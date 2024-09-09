//|
//|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
//|    Authors:  Konstantinos Chatzilygeroudis
//|    email:    costashatz@gmail.com
//|    website:  https://nosalro.github.io/
//|              http://cilab.math.upatras.gr/
//|
//|    This file is part of algevo.
//|
//|    All rights reserved.
//|
//|    Redistribution and use in source and binary forms, with or without
//|    modification, are permitted provided that the following conditions are met:
//|
//|    1. Redistributions of source code must retain the above copyright notice, this
//|       list of conditions and the following disclaimer.
//|
//|    2. Redistributions in binary form must reproduce the above copyright notice,
//|       this list of conditions and the following disclaimer in the documentation
//|       and/or other materials provided with the distribution.
//|
//|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//|
#ifndef ALGEVO_ALGO_UNSTRUCTURED_MAP_ELITES_HPP
#define ALGEVO_ALGO_UNSTRUCTURED_MAP_ELITES_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {

        // Code adapted from https://github.com/hucebot/fast_map-elites
        /// START OF LICENSE
        // BSD 2-Clause License
        // Copyright (c) 2022, HuCeBot Inria/Loria team
        // All rights reserved.
        // Redistribution and use in source and binary forms, with or without
        // modification, are permitted provided that the following conditions are met:
        // 1. Redistributions of source code must retain the above copyright notice, this
        // list of conditions and the following disclaimer.
        // 2. Redistributions in binary form must reproduce the above copyright notice,
        // this list of conditions and the following disclaimer in the documentation
        // and/or other materials provided with the distribution.
        // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        /// END OF LICENSE
        template <typename Fit, typename Scalar = double>
        class UnstructuredMapElites {
        public:
            using mat_t = Eigen::Matrix<Scalar, -1, -1>;
            using x_t = Eigen::Matrix<Scalar, -1, 1>;
            using batch_ranks_t = std::vector<int>;
            using fit_eval_t = std::vector<Fit>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
            using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
            using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

            struct Params {
                int seed = -1;

                unsigned int dim = 0;
                unsigned int dim_features = 0;
                unsigned int pop_size = 0;
                unsigned int num_cells = 0;
                // unsigned int num_cells_mult = 4;
                double min_dist = 0.2;
                double min_dist_min = 1e-05;
                double min_dist_max = 1e+05;

                Scalar exploration_percentage = 0.1;

                Scalar sigma_1 = static_cast<Scalar>(0.005);
                Scalar sigma_2 = static_cast<Scalar>(0.3);

                x_t min_value;
                x_t max_value;

                x_t min_feat;
                x_t max_feat;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;

                unsigned int archive_size = 0;
                std::vector<unsigned int> valid_individuals;
            };

            UnstructuredMapElites(const Params& params) : _params(params), _rgen(0., 1., params.seed), _rgen_ranks(0, params.num_cells - 1, params.seed)
            {
                assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.dim_features > 0 && "Neighbor size not set!");
                assert(_params.num_cells > 0 && "Number of cells not set!");
                assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");
                assert(_params.min_feat.size() == _params.dim_features && _params.max_feat.size() == _params.dim_features && "Min/max feature dimensions should be the same as the feature dimensions!");

                _allocate_data();

                // Initialize random archive
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        Scalar range = (_params.max_value[j] - _params.min_value[j]);
                        _archive(j, i) = _rgen.rand() * range + _params.min_value[j];
                    }
                }
            }

            Params& params() { return _params; }
            const mat_t& population() const { return _archive; }
            mat_t& population() { return _archive; }

            mat_t archive() const
            {
                const unsigned int archive_sz = archive_size();
                mat_t archive(_params.dim, archive_sz);
                if (archive_sz > 0) {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i)) {
                            archive.col(idx++) = _archive.col(i);
                        }
                    }
                }
                return archive;
            }

            // WARNING: This is making the previous log dirty and not to be trusted! It returns a new log!
            IterationLog update_features(const mat_t& features)
            {
                const unsigned int archive_sz = archive_size();
                for (unsigned int i = 0; i < archive_sz; i++) {
                    _archive_features.col(_log.valid_individuals[i]) = features.col(i);
                }

                return _recompute_archive();
            }

            IterationLog update_log(const mat_t& pop, const mat_t& features, const x_t& fit)
            {
                _archive_features = mat_t::Constant(_params.dim_features, _params.num_cells, -std::numeric_limits<Scalar>::max());
                _archive_fit = x_t::Constant(_params.num_cells, -std::numeric_limits<Scalar>::max());
                _archive = mat_t(_params.dim, _params.num_cells);

                _archive_features = features;
                _archive = pop;
                _archive_fit = fit;

                int best_i;
                _log.best_value = _archive_fit.maxCoeff(&best_i);
                _log.best = _archive.col(best_i);
                _log.archive_size = archive_size();
                _log.valid_individuals.resize(_log.archive_size);
                {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i))
                            _log.valid_individuals[idx++] = i;
                    }
                }

                return _log;
            }

            const mat_t& all_features() const { return _archive_features; }
            mat_t& all_features() { return _archive_features; }

            mat_t features() const
            {
                const unsigned int archive_sz = archive_size();
                mat_t features(_params.dim_features, archive_sz);
                if (archive_sz > 0) {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i)) {
                            features.col(idx++) = _archive_features.col(i);
                        }
                    }
                }
                return features;
            }

            std::pair<mat_t, mat_t> archive_features() const
            {
                const unsigned int archive_sz = archive_size();
                mat_t features(_params.dim_features, archive_sz);
                mat_t archive(_params.dim, archive_sz);
                if (archive_sz > 0) {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i)) {
                            features.col(idx) = _archive_features.col(i);
                            archive.col(idx++) = _archive.col(i);
                        }
                    }
                }
                return {archive, features};
            }

            bool valid_individual(unsigned int i) const
            {
                return (_archive_features.col(i).array() != -std::numeric_limits<Scalar>::max()).all();
            }

            const x_t& archive_fit() const { return _archive_fit; }

            Scalar qd_score() const
            {
                Scalar qd = 0;
                for (unsigned int i = 0; i < _params.num_cells; i++)
                    if (_archive_fit(i) != -std::numeric_limits<Scalar>::max())
                        qd += _archive_fit(i);
                return qd;
            }

            unsigned int archive_size() const
            {
                unsigned int sz = 0;
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    if (valid_individual(i))
                        sz++;
                }
                return sz;
            }

            void VAT(int target, Scalar K)
            {
                Scalar max_dist = -std::numeric_limits<Scalar>::max();
                int n = _archive_features.col(0).size();
                for (int i = 0; i < static_cast<int>(_params.num_cells); i++) {
                    if (valid_individual(i)) {
                        for (int j = 0; j < static_cast<int>(_params.num_cells); j++) {
                            if ( i != j && valid_individual(j)) {
                                Scalar d = (_archive_features.col(i) - _archive_features.col(j)).squaredNorm();
                                max_dist = std::max(max_dist, d);
                            }
                        }
                    }
                }
                 _params.min_dist = std::max(_params.min_dist_min, std::min(_params.min_dist_max, max_dist / std::pow((K * target), 1/static_cast<Scalar>(n))));
            }

            void CSC(int target, Scalar K)
            {
                _params.min_dist = _params.min_dist * (1 + (K * (static_cast<int>(archive_size()) - target)));
                _params.min_dist = std::max(_params.min_dist_min, std::min(_params.min_dist_max, _params.min_dist));
            }

            void update_min_dist(int target, Scalar K, const std::string& method = "CSC")
            {
                 if (method == "CSC")
                     CSC(target, K);
                 else if (method == "VAT")
                     VAT(target, K);
            }

            IterationLog step()
            {
                // Uniform random selection
                for (unsigned int i = 0; i < _params.pop_size * 2; i++) {
                    if (_log.valid_individuals.size() > _params.exploration_percentage * _params.num_cells) { // if we have enough filled niches, we select among them
                        unsigned int idx = _log.valid_individuals.size();
                        while (idx >= _log.valid_individuals.size())
                            idx = _rgen_ranks.rand();
                        _batch_ranks[i] = _log.valid_individuals[idx];
                    }
                    else
                        _batch_ranks[i] = _rgen_ranks.rand(); // else we select from all the map!
                    // if not a filled niche, resample to increase exploration
                    if (!valid_individual(_batch_ranks[i])) {
                        for (unsigned int j = 0; j < _params.dim; j++) {
                            Scalar range = (_params.max_value[j] - _params.min_value[j]);
                            _archive(j, _batch_ranks[i]) = _rgen.rand() * range + _params.min_value[j];
                        }
                    }
                }

                // Crossover - line variation
                for (unsigned int i = 0; i < _params.pop_size; i++)
                    _batch.col(i) = _archive.col(_batch_ranks[i * 2]) + _params.sigma_2 * _rgen_gauss.rand() * (_archive.col(_batch_ranks[i * 2]) - _archive.col(_batch_ranks[i * 2 + 1]));

                // Gaussian mutation
                for (unsigned int i = 0; i < _params.pop_size; i++)
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        _batch(j, i) += _rgen_gauss.rand() * _params.sigma_1;
                        // clip in [min,max]
                        _batch(j, i) = std::max(_params.min_value[j], std::min(_params.max_value[j], _batch(j, i)));
                    }

                // evaluate the batch
                tools::parallel_loop(0, _params.pop_size, [this](unsigned int i) {
                    // TO-DO: Check how to avoid copying here
                    x_t p(_params.dim_features);
                    std::tie(_batch_fit(i), p) = _fit_evals[i].eval_qd(_batch.col(i), i);
                    // clip in [min,max]
                    for (unsigned int j = 0; j < _params.dim_features; j++) {
                        p(j) = std::max(_params.min_feat[j], std::min(_params.max_feat[j], p(j)));
                    }
                    _batch_features.col(i) = p;
                });

                // competition
                std::fill(_new_rank.begin(), _new_rank.end(), -1);
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    // search for the closest centroid / the grid
                    bool add_to_container = true;
                    std::vector<int> neighbors;
                    for (int j = 0; j < static_cast<int>(_params.num_cells); j++) {
                        if (valid_individual(j)) {
                            Scalar d = (_batch_features.col(i) - _archive_features.col(j)).squaredNorm();
                            if (d < _params.min_dist) {
                                neighbors.push_back(j);
                                add_to_container = false;
                            }
                        }
                    }

                    if (!add_to_container) {
                        int wins = 0;
                        int best_i = -1;
                        for (unsigned int j = 0; j < neighbors.size(); j++) {
                            if (_ignore_reward || (_batch_fit(i) > _archive_fit(neighbors[j]))) {
                                best_i = neighbors[j];
                                wins++;
                            }
                        }

                        if (wins == static_cast<int>(neighbors.size())) {
                            for (int j = 0; j < static_cast<int>(neighbors.size()); j++) {
                                _archive.col(neighbors[j]) = x_t::Constant(_params.dim, -std::numeric_limits<Scalar>::max());
                                _archive_fit(neighbors[j]) = -std::numeric_limits<Scalar>::max();
                                _archive_features.col(neighbors[j]) = x_t::Constant(_params.dim_features, -std::numeric_limits<Scalar>::max());
                            }
                            _new_rank[i] = best_i;
                        }
                    }
                    else {
                        for (unsigned int j = 0; j < _params.num_cells; j++) {
                            if (!valid_individual(j)) {
                                _new_rank[i] = j;
                                break;
                            }
                        }
                    }
                    if (_new_rank[i] != -1) {
                        _archive.col(_new_rank[i]) = _batch.col(i);
                        _archive_fit(_new_rank[i]) = _batch_fit(i);
                        _archive_features.col(_new_rank[i]) = _batch_features.col(i);
                    }
                }

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size;
                int best_i;
                _log.best_value = _archive_fit.maxCoeff(&best_i);
                _log.best = _archive.col(best_i);
                _log.archive_size = archive_size();
                _log.valid_individuals.resize(_log.archive_size);
                {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i))
                            _log.valid_individuals[idx++] = i;
                    }
                }

                return _log;
            }

            void step_evolution()
            {
                // Uniform random selection
                for (unsigned int i = 0; i < _params.pop_size * 2; i++) {
                    if (_log.valid_individuals.size() > _params.exploration_percentage * _params.num_cells) { // if we have enough filled niches, we select among them
                        unsigned int idx = _log.valid_individuals.size();
                        while (idx >= _log.valid_individuals.size())
                            idx = _rgen_ranks.rand();
                        _batch_ranks[i] = _log.valid_individuals[idx];
                    }
                    else
                        _batch_ranks[i] = _rgen_ranks.rand(); // else we select from all the map!
                    // if not a filled niche, resample to increase exploration
                    if (!valid_individual(_batch_ranks[i])) {
                        for (unsigned int j = 0; j < _params.dim; j++) {
                            Scalar range = (_params.max_value[j] - _params.min_value[j]);
                            _archive(j, _batch_ranks[i]) = _rgen.rand() * range + _params.min_value[j];
                        }
                    }
                }

                // Crossover - line variation
                for (unsigned int i = 0; i < _params.pop_size; i++)
                    _batch.col(i) = _archive.col(_batch_ranks[i * 2]) + _params.sigma_2 * _rgen_gauss.rand() * (_archive.col(_batch_ranks[i * 2]) - _archive.col(_batch_ranks[i * 2 + 1]));

                // Gaussian mutation
                for (unsigned int i = 0; i < _params.pop_size; i++)
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        _batch(j, i) += _rgen_gauss.rand() * _params.sigma_1;
                        // clip in [min,max]
                        _batch(j, i) = std::max(_params.min_value[j], std::min(_params.max_value[j], _batch(j, i)));
                    }

                // evaluate the batch
                tools::parallel_loop(0, _params.pop_size, [this](unsigned int i) {
                    // TO-DO: Check how to avoid copying here
                    x_t p(_params.dim_features);
                    std::tie(_batch_fit(i), p) = _fit_evals[i].eval_qd(_batch.col(i), i);
                });
            }

            IterationLog step_update(mat_t& p)
            {
                tools::parallel_loop(0, _params.pop_size, [this, &p](unsigned int i) {
                    for (unsigned int j = 0; j < _params.dim_features; j++) {
                        // clip in [min,max]
                        p(i, j) = std::max(_params.min_feat[j], std::min(_params.max_feat[j], p(i, j)));
                    }
                    _batch_features.col(i) = p.row(i);
                });

                // competition
                std::fill(_new_rank.begin(), _new_rank.end(), -1);
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    // search for the closest centroid / the grid
                    bool add_to_container = true;
                    std::vector<int> neighbors;
                    for (int j = 0; j < static_cast<int>(_params.num_cells); j++) {
                        if (valid_individual(j)) {
                            Scalar d = (_batch_features.col(i) - _archive_features.col(j)).squaredNorm();
                            if (d < _params.min_dist) {
                                neighbors.push_back(j);
                                add_to_container = false;
                            }
                        }
                    }

                    if (!add_to_container) {
                        int wins = 0;
                        int best_i = -1;
                        for (unsigned int j = 0; j < neighbors.size(); j++) {
                            if (_ignore_reward || (_batch_fit(i) > _archive_fit(neighbors[j]))) {
                            // if ((!_ignore_reward && (_batch_fit(i) > _archive_fit(best_i))) || !valid_individual(best_i))
                                best_i = neighbors[j];
                                wins++;
                            }
                        }

                        if (wins == static_cast<int>(neighbors.size())) {
                            for (int j = 0; j < static_cast<int>(neighbors.size()); j++) {
                                _archive.col(neighbors[j]) = x_t::Constant(_params.dim, -std::numeric_limits<Scalar>::max());
                                _archive_fit(neighbors[j]) = -std::numeric_limits<Scalar>::max();
                                _archive_features.col(neighbors[j]) = x_t::Constant(_params.dim_features, -std::numeric_limits<Scalar>::max());
                            }
                            _new_rank[i] = best_i;
                        }
                    }
                    else {
                        for (unsigned int j = 0; j < _params.num_cells; j++) {
                            if (!valid_individual(j)) {
                                _new_rank[i] = j;
                                break;
                            }
                        }
                    }
                    if (_new_rank[i] != -1) {
                    // if (_new_rank[i] != -1 && ((!_ignore_reward && (_batch_fit(i) > _archive_fit(_new_rank[i]))) || !valid_individual(_new_rank[i]))) {
                        _archive.col(_new_rank[i]) = _batch.col(i);
                        _archive_fit(_new_rank[i]) = _batch_fit(i);
                        _archive_features.col(_new_rank[i]) = _batch_features.col(i);
                    }
                }

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size;
                int best_i;
                _log.best_value = _archive_fit.maxCoeff(&best_i);
                _log.best = _archive.col(best_i);
                _log.archive_size = archive_size();
                _log.valid_individuals.resize(_log.archive_size);
                {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i))
                            _log.valid_individuals[idx++] = i;
                    }
                }

                return _log;
            }

            void ignore_reward(bool v)
            {
                _ignore_reward = v;
            }

        protected:
            // Parameters
            Params _params;

            // Iteration Log
            IterationLog _log;

            // Actual population (current values)
            mat_t _archive;
            x_t _archive_fit;
            mat_t _archive_features;

            // Centroids

            // Batch
            mat_t _batch;
            mat_t _batch_features;
            x_t _batch_fit;
            batch_ranks_t _batch_ranks;
            batch_ranks_t _new_rank;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen;
            tools::rgen_int_t _rgen_ranks;
            rgen_scalar_gauss_t _rgen_gauss = rgen_scalar_gauss_t(static_cast<Scalar>(0.), static_cast<Scalar>(1.));
            bool _ignore_reward = false;

            void _allocate_data()
            {
                // _params.num_cells = _params.num_cells_mult * _params.num_cells;
                _archive = mat_t(_params.dim, _params.num_cells);
                _archive_features = mat_t::Constant(_params.dim_features, _params.num_cells, -std::numeric_limits<Scalar>::max());
                _archive_fit = x_t::Constant(_params.num_cells, -std::numeric_limits<Scalar>::max());

                _batch = mat_t(_params.dim, _params.pop_size);
                _batch_fit = x_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());
                _batch_features = mat_t(_params.dim_features, _params.pop_size);

                _batch_ranks.resize(_params.pop_size * 2);
                _new_rank.resize(_params.pop_size);

                _fit_evals.resize(_params.pop_size);
            }

            IterationLog _recompute_archive()
            {
                // Let's reassign all archive
                batch_ranks_t ranks(_params.num_cells, -1);
                tools::parallel_loop(0, _params.num_cells, [this, &ranks](unsigned int i) {
                    if (!valid_individual(i))
                        return;
                    // search for the closest centroid / the grid
                    int best_i = -1;
                    // TO-DO: Do not iterate over all cells; make a tree or something
                    // Scalar best_dist = std::numeric_limits<Scalar>::max();
                    // for (int j = 0; j < static_cast<int>(_params.num_cells); j++) {
                    //     Scalar d = (_archive_features.col(i) - _centroids.col(j)).squaredNorm();
                    //     if (d < best_dist) {
                    //         best_dist = d;
                    //         best_i = j;
                    //     }
                    // }

                    // This is the same as the for loop, but shorter
                    // (_centroids.colwise() - _archive_features.col(i)).colwise().squaredNorm().minCoeff(&best_i);

                    // We do not want to check fitness here (just re-assigning cells)
                    ranks[i] = best_i;
                });

                // apply the new ranks
                // We need to copy everything and re-fill
                mat_t old_archive = _archive;
                x_t old_archive_fit = _archive_fit;
                mat_t old_archive_features = _archive_features;

                // re-initialize features/fitness
                _archive_features = mat_t::Constant(_params.dim_features, _params.num_cells, -std::numeric_limits<Scalar>::max());
                _archive_fit = x_t::Constant(_params.num_cells, -std::numeric_limits<Scalar>::max());

                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    if (ranks[i] != -1 && (_ignore_reward || (old_archive_fit(i) > _archive_fit(ranks[i])))) {
                    // if (_new_rank[i] != -1 && ((!_ignore_reward && (_batch_fit(i) > _archive_fit(ranks[i]))) || !valid_individual(ranks[i]))) {
                        _archive.col(ranks[i]) = old_archive.col(i);
                        _archive_fit(ranks[i]) = old_archive_fit(i);
                        _archive_features.col(ranks[i]) = old_archive_features.col(i);
                    }
                }

                int best_i;
                _log.best_value = _archive_fit.maxCoeff(&best_i);
                _log.best = _archive.col(best_i);
                _log.archive_size = archive_size();
                _log.valid_individuals.resize(_log.archive_size);
                {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i))
                            _log.valid_individuals[idx++] = i;
                    }
                }

                return _log;
            }
        };
    } // namespace algo
} // namespace algevo

#endif
