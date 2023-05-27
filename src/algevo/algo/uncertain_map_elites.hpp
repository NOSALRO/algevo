//|
//|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
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
#ifndef ALGEVO_ALGO_UNCERTAIN_MAP_ELITES_HPP
#define ALGEVO_ALGO_UNCERTAIN_MAP_ELITES_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        // TO-DO: This need re-implementation if the descriptors are also noisy!
        // aka: clean the archive and re-insert them
        template <typename Fit, typename Scalar = double>
        class UncertainMapElites {
        public:
            using mat_t = Eigen::Matrix<Scalar, -1, -1>;
            using x_t = Eigen::Matrix<Scalar, -1, 1>;
            using batch_ranks_t = std::vector<int>;
            using fit_eval_t = std::vector<Fit>;
            using archive_t = std::vector<mat_t>;

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

                unsigned int cell_depth = 1;
                bool re_evaluate = false;
                Scalar forgetting_factor = 0.5;

                Scalar exploration_percentage = 0.1;

                Scalar sigma_1 = static_cast<Scalar>(0.005);
                Scalar sigma_2 = static_cast<Scalar>(0.3);

                x_t min_value;
                x_t max_value;

                x_t min_feat;
                x_t max_feat;

                mat_t centroids;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;

                unsigned int archive_size = 0;
                std::vector<unsigned int> valid_individuals;
            };

            UncertainMapElites(const Params& params) : _params(params), _rgen(0., 1., params.seed), _rgen_ranks(0, params.num_cells - 1, params.seed)
            {
                assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.dim_features > 0 && "Neighbor size not set!");
                assert(_params.num_cells > 0 && "Number of cells not set!");
                assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");
                assert(_params.min_feat.size() == _params.dim_features && _params.max_feat.size() == _params.dim_features && "Min/max feature dimensions should be the same as the feature dimensions!");
                assert((_params.centroids.size() == 0 || (_params.centroids.rows() == _params.dim_features && _params.centroids.cols() == _params.num_cells)) && "Centroids dimensions not set correctly!");

                _allocate_data();

                // Initialize random archive
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        Scalar range = (_params.max_value[j] - _params.min_value[j]);
                        _archive[i](j, 0) = _rgen.rand() * range + _params.min_value[j];
                    }
                }

                // Initialize random centroids if needed
                if (_params.centroids.size() == 0) {
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        for (unsigned int j = 0; j < _params.dim_features; j++) {
                            Scalar range = (_params.max_feat[j] - _params.min_feat[j]);
                            _centroids(j, i) = _rgen.rand() * range + _params.min_feat[j];
                        }
                    }
                }
                else {
                    _centroids = _params.centroids;
                }
            }

            mat_t population() const
            {
                mat_t pop(_params.dim, _params.num_cells);
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    pop.col(i) = _archive[i].col(_best[i]);
                }

                return pop;
            }

            mat_t archive() const
            {
                const unsigned int archive_sz = archive_size();
                mat_t archive(_params.dim, archive_sz);
                if (archive_sz > 0) {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i)) {
                            archive.col(idx++) = _archive[i].col(_best[i]);
                        }
                    }
                }
                return archive;
            }

            const mat_t& centroids() const { return _centroids; }
            mat_t& centroids() { return _centroids; }

            mat_t all_features() const
            {
                mat_t features(_params.dim_features, _params.num_cells);
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    features.col(i) = _archive_features[i].col(_best[i]);
                }

                return features;
            }

            mat_t features() const
            {
                const unsigned int archive_sz = archive_size();
                mat_t features(_params.dim_features, archive_sz);
                if (archive_sz > 0) {
                    unsigned int idx = 0;
                    for (unsigned int i = 0; i < _params.num_cells; i++) {
                        if (valid_individual(i)) {
                            features.col(idx++) = _archive_features[i].col(_best[i]);
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
                            features.col(idx) = _archive_features[i].col(_best[i]);
                            archive.col(idx++) = _archive[i].col(_best[i]);
                        }
                    }
                }
                return {archive, features};
            }

            bool valid_individual(unsigned int i, unsigned int depth = 0) const
            {
                return (_archive_fit(i, depth) != -std::numeric_limits<Scalar>::max());
            }

            x_t archive_fit() const
            {
                x_t fit(_params.num_cells);
                for (unsigned int i = 0; i < _params.num_cells; i++)
                    fit[i] = _archive_fit(i, _best[i]);
                return fit;
            }

            Scalar qd_score() const
            {
                Scalar qd = 0;
                for (unsigned int i = 0; i < _params.num_cells; i++)
                    if (_archive_fit(i, _best[i]) != -std::numeric_limits<Scalar>::max())
                        qd += _archive_fit(i, _best[i]);
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

            unsigned int total_size() const
            {
                unsigned int sz = 0;
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    for (unsigned int d = 0; d < _params.cell_depth; d++)
                        if (valid_individual(i, d))
                            sz++;
                }
                return sz;
            }

            IterationLog step()
            {
                unsigned int extra_evals = 0;
                if (_log.iterations > 0 && _params.re_evaluate) {
                    // revaluate all population
                    tools::parallel_loop(0, _params.num_cells * _params.cell_depth, [this](unsigned int i) {
                        unsigned int cell = i / _params.cell_depth;
                        unsigned int depth = i % _params.cell_depth;

                        if (_archive_fit(cell, depth) == -std::numeric_limits<Scalar>::max()) { // no need to re-evaluate empty slot
                            return;
                        }

                        // TO-DO: Check how to avoid copying here
                        x_t p(_params.dim_features);
                        Scalar v;
                        std::tie(v, p) = _fit_evals[i].eval_qd(_archive[cell].col(depth));
                        // clip in [min,max]
                        for (unsigned int j = 0; j < _params.dim_features; j++) {
                            p(j) = std::max(_params.min_feat[j], std::min(_params.max_feat[j], p(j)));
                        }

                        _archive_fit(cell, depth) += _params.forgetting_factor * (v - _archive_fit(cell, depth));
                        _archive_features[cell].col(depth).array() += _params.forgetting_factor * (p.array() - _archive_features[cell].col(depth).array());
                    });

                    extra_evals = total_size();

                    // Recompute new per cell ranks
                    _recompute_ranks();
                }

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
                            _archive[_batch_ranks[i]](j, _best[_batch_ranks[i]]) = _rgen.rand() * range + _params.min_value[j];
                        }
                    }
                }

                // Crossover - line variation
                for (unsigned int i = 0; i < _params.pop_size; i++)
                    _batch.col(i) = _archive[_batch_ranks[i * 2]].col(_best[_batch_ranks[i * 2]]) + _params.sigma_2 * _rgen_gauss.rand() * (_archive[_batch_ranks[i * 2]].col(_best[_batch_ranks[i * 2]]) - _archive[_batch_ranks[i * 2 + 1]].col(_best[_batch_ranks[i * 2 + 1]]));

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
                    std::tie(_batch_fit(i), p) = _fit_evals[i].eval_qd(_batch.col(i));
                    // clip in [min,max]
                    for (unsigned int j = 0; j < _params.dim_features; j++) {
                        p(j) = std::max(_params.min_feat[j], std::min(_params.max_feat[j], p(j)));
                    }
                    _batch_features.col(i) = p;
                });

                // competition
                std::fill(_new_rank.begin(), _new_rank.end(), -1);
                tools::parallel_loop(0, _params.pop_size, [this](unsigned int i) {
                    // search for the closest centroid / the grid
                    int best_i = -1;
                    // TO-DO: Do not iterate over all cells; make a tree or something
                    Scalar best_dist = std::numeric_limits<Scalar>::max();
                    for (int j = 0; j < static_cast<int>(_params.num_cells); j++) {
                        Scalar d = (_batch_features.col(i) - _centroids.col(j)).squaredNorm();
                        if (d < best_dist) {
                            best_dist = d;
                            best_i = j;
                        }
                    }

                    // This is the same as the for loop, but shorter
                    // (_centroids.colwise() - _batch_features.col(i)).colwise().squaredNorm().minCoeff(&best_i);

                    if (_batch_fit(i) > _archive_fit(best_i, _worst[best_i]))
                        _new_rank[i] = best_i;
                });

                batch_ranks_t to_reevaluate;

                // apply the new ranks
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_new_rank[i] != -1) {
                        to_reevaluate.push_back(_new_rank[i]);
                        _replace(_new_rank[i], _batch.col(i), _batch_fit(i), _batch_features.col(i));
                    }
                }

                // Recompute new per cell ranks
                _recompute_ranks();

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size + extra_evals;
                unsigned int best_i;
                std::tie(best_i, _log.best_value) = _get_best();
                _log.best = _archive[best_i].col(_best[best_i]);
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

            void _replace(unsigned int index, const x_t& x, Scalar val, const x_t& f)
            {
                int replace_i = 0;
                Scalar replace_val = std::numeric_limits<Scalar>::max();
                for (unsigned int i = 0; i < _params.cell_depth; i++) {
                    // We found an empty cell
                    if (_archive_fit(index, i) == -std::numeric_limits<Scalar>::max()) {
                        replace_i = i;
                        break;
                    }
                    else if (_archive_fit(index, i) < replace_val) {
                        replace_i = i;
                        replace_val = _archive_fit(index, i);
                    }
                }

                _archive[index].col(replace_i) = x;
                _archive_fit(index, replace_i) = val;
                _archive_features[index].col(replace_i) = f;
            }

            void _recompute_ranks()
            {
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    for (unsigned int j = 0; j < _params.cell_depth; j++) {
                        if (_archive_fit(i, j) > _archive_fit(i, _best[i]))
                            _best[i] = j;
                        else if (_archive_fit(i, j) < _archive_fit(i, _worst[i]))
                            _worst[i] = j;
                    }
                }
            }

            std::pair<unsigned int, Scalar> _get_best() const
            {
                unsigned int best_i = 0;
                Scalar best_value = -std::numeric_limits<Scalar>::max();
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    if (_archive_fit(i, _best[i]) > best_value) {
                        best_value = _archive_fit(i, _best[i]);
                        best_i = i;
                    }
                }

                return {best_i, best_value};
            }

        protected:
            // Parameters
            Params _params;

            // Iteration Log
            IterationLog _log;

            // Population
            archive_t _archive;
            mat_t _archive_fit;
            archive_t _archive_features;
            batch_ranks_t _best, _worst;

            // Centroids
            mat_t _centroids;

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

            void _allocate_data()
            {
                _archive.resize(_params.num_cells);
                _archive_features.resize(_params.num_cells);
                for (unsigned int i = 0; i < _params.num_cells; i++) {
                    _archive[i] = mat_t(_params.dim, _params.cell_depth);
                    _archive_features[i] = mat_t::Constant(_params.dim_features, _params.cell_depth, -std::numeric_limits<Scalar>::max());
                }
                _archive_fit = mat_t::Constant(_params.num_cells, _params.cell_depth, -std::numeric_limits<Scalar>::max());

                _best.resize(_params.num_cells, 0);
                _worst.resize(_params.num_cells, _params.cell_depth);

                _batch = mat_t(_params.dim, _params.pop_size);
                _batch_fit = x_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());
                _batch_features = mat_t(_params.dim_features, _params.pop_size);

                _centroids = mat_t(_params.dim_features, _params.num_cells);

                _batch_ranks.resize(_params.pop_size * 2);
                _new_rank.resize(_params.pop_size);

                _fit_evals.resize(std::max(_params.pop_size, _params.num_cells * _params.cell_depth));
            }
        };
    } // namespace algo
} // namespace algevo

#endif
