#ifndef ALGEVO_ALGO_MAP_ELITES_HPP
#define ALGEVO_ALGO_MAP_ELITES_HPP

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
        template <typename Params, typename Fit, typename Scalar = double>
        class MapElites {
        public:
            using archive_t = Eigen::Matrix<Scalar, -1, -1>;
            using centroids_t = Eigen::Matrix<Scalar, -1, -1>;
            using batch_t = Eigen::Matrix<Scalar, -1, -1>;
            using fit_t = Eigen::Matrix<Scalar, -1, 1>;
            using x_t = Eigen::Matrix<Scalar, 1, -1>;

            using batch_ranks_t = std::array<int, Params::batch_size * 2>;
            using new_rank_t = std::array<int, Params::batch_size>;

            using fit_eval_t = std::array<Fit, Params::batch_size>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
            using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
            using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

            MapElites()
            {
                _allocate_data();

                if (Params::grid) {
                    // TO-DO: Allow for bigger grids
                    static_assert((Params::grid && Params::dim_features == 2) || !Params::grid, "Too big of a grid!");
                    for (unsigned int i = 0; i < Params::grid_size; i++)
                        for (unsigned int j = 0; j < Params::grid_size; j++) {
                            _centroids(i * Params::grid_size + j, 0) = static_cast<Scalar>(i) / Params::grid_size;
                            _centroids(i * Params::grid_size + j, 1) = static_cast<Scalar>(j) / Params::grid_size;
                        }
                }

                // Initialize random archive
                for (unsigned int i = 0; i < Params::num_cells; i++) {
                    for (unsigned int j = 0; j < Params::dim; j++) {
                        _archive(i, j) = _rgen.rand();
                    }
                }

                // Initialize random centroids
                for (unsigned int i = 0; i < Params::num_cells; i++) {
                    for (unsigned int j = 0; j < Params::dim_features; j++) {
                        _centroids(i, j) = _rgen_features.rand();
                    }
                }
            }

            const archive_t& population() const { return _archive; }
            archive_t& population() { return _archive; }

            const archive_t& archive() const { return _archive; }
            archive_t& archive() { return _archive; }

            const centroids_t& centroids() const { return _centroids; }
            centroids_t& centroids() { return _centroids; }

            const fit_t& archive_fit() const { return _archive_fit; }

            double qd_score() const
            {
                double qd = 0;
                for (int i = 0; i < Params::num_cells; i++)
                    if (_archive_fit(i) != -std::numeric_limits<Scalar>::max())
                        qd += _archive_fit(i);
                return qd;
            }

            void step()
            {
                // Uniform random selection
                for (unsigned int i = 0; i < Params::batch_size * 2; i++)
                    _batch_ranks[i] = _rgen_ranks.rand(); // yes, from all the map!

                // Crossover - line variation
                for (unsigned int i = 0; i < Params::batch_size; i++)
                    _batch.row(i) = _archive.row(_batch_ranks[i * 2]) + Params::sigma_2 * _rgen_gauss.rand() * (_archive.row(_batch_ranks[i * 2]) - _archive.row(_batch_ranks[i * 2 + 1]));

                // Gaussian mutation
                for (unsigned int i = 0; i < Params::batch_size; i++)
                    for (unsigned int j = 0; j < Params::dim; j++)
                        _batch(i, j) += _rgen_gauss.rand() * Params::sigma_1;
                // clip in [min,max] -- TO-DO: Maybe do this inside the mutation, to remove extra loop
                for (unsigned int i = 0; i < Params::batch_size; ++i)
                    _batch.row(i) = _batch.row(i).cwiseMin(Params::max_value).cwiseMax(Params::min_value);

                // evaluate the batch
                tools::parallel_loop(0, Params::batch_size, [this](unsigned int i) {
                    // TO-DO: Check how to avoid copying here
                    x_t p(Params::dim_features);
                    std::tie(_batch_fit(i), p) = _fit_evals[i].eval(_batch.row(i));
                    _batch_features.row(i) = p.cwiseMin(Params::max_features_value).cwiseMax(Params::min_features_value);
                });

                // competition
                std::fill(_new_rank.begin(), _new_rank.end(), -1);
                tools::parallel_loop(0, Params::batch_size, [this](unsigned int i) {
                    // search for the closest centroid / the grid
                    int best_i = -1;
                    if (Params::grid) {
                        int x = std::round(_batch_features(i, 0) * (Params::grid_size - 1));
                        int y = std::round(_batch_features(i, 1) * (Params::grid_size - 1));
                        best_i = x * Params::grid_size + y;
                    }
                    else {
                        // TO-DO: Do not iterate over all cells; make a tree or something
                        double best_dist = std::numeric_limits<Scalar>::max();
                        for (int j = 0; j < static_cast<int>(Params::num_cells); j++) {
                            double d = (_batch_features.row(i) - _centroids.row(j)).squaredNorm();
                            if (d < best_dist) {
                                best_dist = d;
                                best_i = j;
                            }
                        }
                    }
                    if (_batch_fit.row(i)[0] > _archive_fit(best_i))
                        _new_rank[i] = best_i;
                });

                // apply the new ranks
                for (unsigned int i = 0; i < Params::batch_size; i++) {
                    if (_new_rank[i] != -1) {
                        _archive.row(_new_rank[i]) = _batch.row(i);
                        _archive_fit(_new_rank[i]) = _batch_fit(i);
                    }
                }
            }

        protected:
            // Actual population (current values)
            archive_t _archive;
            fit_t _archive_fit;

            // Centroids
            centroids_t _centroids;

            // Batch
            batch_t _batch;
            centroids_t _batch_features;
            fit_t _batch_fit;
            batch_ranks_t _batch_ranks;
            new_rank_t _new_rank;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen = rgen_scalar_t(Params::min_value, Params::max_value, Params::seed);
            rgen_scalar_t _rgen_features = rgen_scalar_t(Params::min_features_value, Params::max_features_value, Params::seed);
            tools::rgen_int_t _rgen_ranks = tools::rgen_int_t(0, Params::num_cells - 1, Params::seed);
            rgen_scalar_gauss_t _rgen_gauss = rgen_scalar_gauss_t(0., 1.);

            void _allocate_data()
            {
                _archive = archive_t(Params::num_cells, Params::dim);
                _archive_fit = fit_t::Constant(Params::num_cells, -std::numeric_limits<Scalar>::max());

                _batch = batch_t(Params::batch_size, Params::dim);
                _batch_fit = fit_t::Constant(Params::batch_size, -std::numeric_limits<Scalar>::max());
                _batch_features = centroids_t(Params::batch_size, Params::dim_features);

                _centroids = centroids_t(Params::num_cells, Params::dim_features);
            }
        };
    } // namespace algo
} // namespace algevo

#endif
