#ifndef ALGEVO_TOOLS_CVT_HPP
#define ALGEVO_TOOLS_CVT_HPP

#include <algorithm> //for random_shuffle
#include <numeric> //for iota
#include <vector>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <Eigen/Core>

#include <algevo/tools/random.hpp>

namespace algevo {
    namespace tools {
        // Code adapted from: https://github.com/resibots/cvt
        /// Under GNU General Public License v3.0
        template <typename Scalar = double>
        struct SampledInitializer {
            using Matrix = Eigen::Matrix<Scalar, -1, -1>;

            static Matrix init(const Matrix& data, const unsigned int num_clusters)
            {
                // number of instances = data.cols()
                assert((int)num_clusters <= data.cols());

                std::vector<unsigned int> indices(data.cols());
                std::iota(indices.begin(), indices.end(), 0);
                std::random_shuffle(indices.begin(), indices.end());

                // Create the centroids
                unsigned int dim = data.rows();
                Matrix centroids = Matrix::Zero(dim, num_clusters);

                for (unsigned int i = 0; i < num_clusters; i++)
                    centroids.col(i) = data.col(indices[i]);

                return centroids;
            }
        };

        template <typename Scalar = double>
        struct EuclideanDistance {
            using Vector = Eigen::Matrix<Scalar, 1, -1>;

            static Scalar evaluate(const Vector& p1, const Vector& p2)
            {
                return (p1 - p2).norm();
            }
        };

        template <typename DistanceMetric = EuclideanDistance<>, typename Initializer = SampledInitializer<>, typename Scalar = double>
        class KMeans {
        public:
            using Matrix = Eigen::Matrix<Scalar, -1, -1>;

            KMeans(unsigned int max_iterations = 100, unsigned int restarts = 1, const Scalar tolerance = 1e-8) : _max_iterations(max_iterations), _restarts(restarts), _tolerance(tolerance) {}

            const Matrix& cluster(const Matrix& data, unsigned int num_clusters)
            {
                std::vector<Matrix> all_centroids(_restarts, Matrix::Zero(data.rows(), num_clusters));
                std::vector<Scalar> all_losses(_restarts, 0.0);

                for (unsigned int r = 0; r < _restarts; r++) {
                    // Initialize
                    all_centroids[r] = Initializer::init(data, num_clusters);

                    // Iterate (EM)
                    Scalar loss, prev_loss;
                    loss = prev_loss = 0.0;
                    Scalar delta = _tolerance;

                    for (unsigned int i = 0; i < _max_iterations; i++) {
                        Matrix new_centroids = Matrix::Zero(data.rows(), num_clusters);

                        // Calculate the distances
                        std::vector<unsigned int> counts(num_clusters, 0);
                        loss = _calc_distances(data, all_centroids[r], new_centroids, counts);

                        delta = std::abs(prev_loss - loss);

                        if (delta < _tolerance) {
                            break;
                        }

                        prev_loss = loss;

                        // Update the centroids
                        _update_centroids(new_centroids, counts);

                        all_centroids[r] = new_centroids;
                    }

                    // Store this centroid and the loss
                    all_losses[r] = loss;
                }

                // Return the centroids with the lowest loss
                unsigned int argmin_index = std::distance(all_losses.begin(), std::min_element(all_losses.begin(), all_losses.end()));

                _centroids = all_centroids[argmin_index];

                return _centroids;
            }

        protected:
            Scalar _calc_distances(const Matrix& data, const Matrix& centroids, Matrix& new_centroids, std::vector<unsigned int>& counts)
            {
                unsigned int nb_points = data.cols();
                Scalar sum = 0.;
#ifdef USE_TBB
                static tbb::mutex sm;
#endif

                tools::parallel_loop(0, nb_points, [&](unsigned int i) {
                    // Find the closest centroid to this point.
                    Scalar min_distance = std::numeric_limits<Scalar>::infinity();
                    unsigned int closest_cluster = centroids.cols(); // Invalid value.

                    for (int j = 0; j < centroids.cols(); j++) {
                        const Scalar distance = DistanceMetric::evaluate(data.col(i), centroids.col(j));

                        if (distance < min_distance) {
                            min_distance = distance;
                            closest_cluster = j;
                        }

                        // Since the minimum distance cannot be less than 0
                        // we could accelerate computation by breaking
                        if (min_distance == 0.0)
                            break;
                    }

#ifdef USE_TBB
                    tbb::mutex::scoped_lock lock; // create a lock
                    lock.acquire(sm);
#endif
                    sum += min_distance;
                    // We now have the minimum distance centroid index.
                    new_centroids.col(closest_cluster) += data.col(i);
                    counts[closest_cluster]++;
#ifdef USE_TBB
                    lock.release();
#endif
                });

                // The loss is the mean
                return sum / static_cast<Scalar>(nb_points);
            }

            void _update_centroids(Matrix& new_centroids, const std::vector<unsigned int>& counts)
            {
                // TODO: vectorize
                for (int i = 0; i < new_centroids.cols(); i++) {
                    new_centroids.col(i) = new_centroids.col(i) / (Scalar)counts[i];
                }
            }

            unsigned int _max_iterations;
            unsigned int _restarts;
            Scalar _eta;
            Scalar _tolerance;
            Matrix _centroids;
        };
    } // namespace tools
} // namespace algevo

#endif
