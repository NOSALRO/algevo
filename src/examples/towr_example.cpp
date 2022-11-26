#include <iostream>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <towr/nlp_formulation.h>
#include <towr/terrain/examples/height_map_examples.h>

#include <algevo/algo/pso_grad.hpp>

using namespace towr;
using namespace ifopt;

template <typename Scalar = double>
struct TowrFit {
    static constexpr unsigned int dim = 339;
    static constexpr unsigned int dim_features = dim;
    static constexpr double max_value = 20000.;
    static constexpr double min_value = -20000.;
    static constexpr double max_features_value = 1.;
    static constexpr double min_features_value = 0.;
    static constexpr unsigned int nin_dim = 225;
    static constexpr unsigned int neq_dim = 291;

    using Vector = Eigen::Matrix<Scalar, -1, 1>;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;

    NlpFormulation formulation;
    ifopt::Problem nlp;
    SplineHolder solution;

    TowrFit()
    {
        // terrain
        formulation.terrain_ = std::make_shared<FlatGround>(0.0);

        // Kinematic limits and dynamic parameters of the hopper
        formulation.model_ = RobotModel(RobotModel::Monoped);

        // set the initial position of the hopper
        formulation.initial_base_.lin.at(towr::kPos).z() = 0.5;
        formulation.initial_ee_W_.push_back(Eigen::Vector3d::Zero());

        // define the desired goal state of the hopper
        formulation.final_base_.lin.at(towr::kPos) << 1.0, 0.0, 0.5;

        // Parameters that define the motion. See c'tor for default values or
        // other values that can be modified.
        // First we define the initial phase durations, that can however be changed
        // by the optimizer. The number of swing and stance phases however is fixed.
        // alternating stance and swing:     ____-----_____-----_____-----_____
        formulation.params_.ee_phase_durations_.push_back(
            {0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.2});
        formulation.params_.ee_in_contact_at_start_.push_back(true);

        // Initialize the nonlinear-programming problem with the variables,
        // constraints and costs.

        // formulation.params_.constraints_.erase(formulation.params_.constraints_.begin() + 1);
        formulation.params_.constraints_.clear();
        formulation.params_.constraints_.push_back(towr::Parameters::ConstraintName::Dynamic);
        formulation.params_.constraints_.push_back(towr::Parameters::ConstraintName::Force);
        formulation.params_.constraints_.push_back(towr::Parameters::ConstraintName::Terrain);
        formulation.params_.constraints_.push_back(towr::Parameters::ConstraintName::EndeffectorRom);
        formulation.params_.constraints_.push_back(towr::Parameters::ConstraintName::Swing);
        formulation.params_.constraints_.push_back(towr::Parameters::ConstraintName::BaseAcc);

        for (auto c : formulation.GetVariableSets(solution))
            nlp.AddVariableSet(c);
        for (auto c : formulation.GetConstraints(solution))
            nlp.AddConstraintSet(c);
        for (auto c : formulation.GetCosts())
            nlp.AddCostSet(c);
    }

    std::tuple<Scalar, Vector, Vector, Matrix, Vector, Matrix, Scalar> eval_all(const Vector& x, int verbose = 0)
    {
        // Set variables
        nlp.SetVariables(x.data());

        // Get objective and gradient
        Scalar f = nlp.EvaluateCostFunction(x.data());
        Vector grad = nlp.EvaluateCostFunctionGradient(x.data());

        Vector c(neq_dim);
        c.setZero();
        Matrix C(neq_dim, dim); // equality
        C.setZero();
        Vector g(nin_dim);
        g.setZero();
        Matrix G(nin_dim, dim); // inequality
        G.setZero();

        size_t next_eq = 0;
        size_t next_in = 0;
        // Iterate over constraints to create matrices and violations
        // for (const auto& ct : nlp.GetConstraints().GetComponents()) {
        //     int n = ct->GetRows();
        //     // if (verbose > 0)
        //     //     std::cout << ct->GetName() << ": " << n << std::endl;
        //     if (n < 0)
        //         continue;
        //     // if (ct->GetName() != "rangeofmotion-0")
        //     //     continue;
        //     // if (ct->GetName() != "force-ee-force_0" && ct->GetName() != "splineacc-base-lin" && ct->GetName() != "splineacc-base-ang" && ct->GetName() != "terrain-ee-motion_0" && ct->GetName() != "swing-ee-motion_0")
        //     //     continue;
        //     Vector vals = ct->GetValues();
        //     Matrix cons = ct->GetJacobian();
        //     for (int i = 0; i < n; i++) {
        //         auto bounds = ct->GetBounds()[i];
        //         if (std::abs(bounds.lower_ - bounds.upper_) < 1e-8) {
        //             if (verbose > 1)
        //                 std::cout << "Equality(" << next_eq << "): " << vals[i] << " == " << bounds.upper_ << std::endl;
        //             c[next_eq] = vals[i] - bounds.lower_;
        //             C.row(next_eq) = cons.row(i);

        //             next_eq++;
        //         }
        //         else {
        //             if (verbose > 1)
        //                 std::cout << "Inequality(" << next_in << "): " << vals[i] << " >= " << bounds.lower_ << std::endl;
        //             g[next_in] = vals[i] - bounds.lower_;
        //             G.row(next_eq) = cons.row(i);

        //             if (verbose > 1)
        //                 std::cout << "Inequality(" << next_in + 1 << "): " << -vals[i] << " <= " << bounds.upper_ << std::endl;
        //             g[next_in + 1] = bounds.upper_ - vals[i];
        //             G.row(next_eq) = -cons.row(i);

        //             next_in += 2;
        //         }
        //     }
        // }

        Vector cons = nlp.EvaluateConstraints(x.data());
        const auto& cbounds = nlp.GetBoundsOnConstraints();
        Matrix jac = nlp.GetJacobianOfConstraints();
        for (size_t i = 0; i < cons.size(); i++) {
            if (std::abs(cbounds[i].lower_ - cbounds[i].upper_) < 1e-8) {
                if (verbose > 1) {
                    std::cout << "Equality(" << next_eq << "): " << cons[i] << " == " << cbounds[i].upper_ << std::endl;
                }
                c[next_eq] = cons[i] - cbounds[i].lower_;
                C.row(next_eq) = jac.row(i);
                next_eq++;
            }
            else {
                if (cbounds[i].lower_ > -1e20) {
                    if (verbose > 1)
                        std::cout << "Inequality(" << next_in << "): " << cons[i] << " >= " << cbounds[i].lower_ << std::endl;
                    g[next_in] = cons[i] - cbounds[i].lower_;
                    G.row(next_in) = jac.row(i);
                    next_in++;
                }

                if (cbounds[i].upper_ < 1e20) {
                    if (verbose > 1)
                        std::cout << "Inequality(" << next_in << "): " << -cons[i] << " <= " << cbounds[i].upper_ << std::endl;
                    g[next_in] = cbounds[i].upper_ - cons[i];
                    G.row(next_in) = -jac.row(i);
                    next_in++;
                }
                // next_in += 2;
            }
        }

        // Add variable bounds
        const auto& bounds = nlp.GetBoundsOnOptimizationVariables();
        for (size_t i = 0; i < dim; i++) {
            if (std::abs(bounds[i].lower_ - bounds[i].upper_) < 1e-8) {
                if (verbose > 2) {
                    std::cout << "Equality(" << next_eq << "): " << x[i] << " == " << bounds[i].upper_ << std::endl;
                }
                c[next_eq] = x[i] - bounds[i].lower_;
                C(next_eq, i) = 1.;
                next_eq++;
            }
            else {
                if (bounds[i].lower_ > -1e20) {
                    if (verbose > 2)
                        std::cout << "Inequality(" << next_in << "): " << x[i] << " >= " << bounds[i].lower_ << std::endl;
                    g[next_in] = x[i] - bounds[i].lower_;
                    G(next_in, i) = 1.;
                    next_in++;
                }

                if (bounds[i].upper_ < 1e20) {
                    if (verbose > 2)
                        std::cout << "Inequality(" << next_in << "): " << -x[i] << " <= " << bounds[i].upper_ << std::endl;
                    g[next_in] = bounds[i].upper_ - x[i];
                    G(next_in, i) = -1.;
                    next_in++;
                }

                // next_in += 2;
            }
        }

        // if (verbose > 0) {
        //     for (auto& vv : nlp.GetOptVariables()->GetComponents()) {
        //         int n = vv->GetRows();
        //         const auto& bounds = vv->GetBounds();
        //         std::cout << vv->GetName() << ": " << std::endl;
        //         for (int i = 0; i < n; i++) {
        //             std::cout << " " << i << "(" << vv->GetValues()[i] << ") -> [" << bounds[i].lower_ << ", " << bounds[i].upper_ << "]" << std::endl;
        //         }
        //     }
        // }

        // std::cout << "Got constraints" << std::endl;

        Scalar cv = 0.;
        for (unsigned int i = 0; i < neq_dim; i++) {
            cv += std::abs(c[i]);
        }
        for (unsigned int i = 0; i < nin_dim; i++) {
            cv += std::abs(std::min(0., g[i]));
        }

        if (verbose > 0) {
            // Can directly view the optimization variables through:
            // Eigen::VectorXd x = nlp.GetVariableValues()
            // However, it's more convenient to access the splines constructed from these
            // variables and query their values at specific times:
            using namespace std;
            cout.precision(2);
            nlp.PrintCurrent(); // view variable-set, constraint violations, indices,...
            cout << fixed;

            std::cout << f << " -> " << cv << std::endl;
            // cout << "\n====================\nMonoped trajectory:\n====================\n";

            // double t = 0.0;
            // while (t <= solution.base_linear_->GetTotalTime() + 1e-5) {
            //     cout << "t=" << t << "\n";
            //     cout << "Base linear position x,y,z:   \t";
            //     cout << solution.base_linear_->GetPoint(t).p().transpose() << "\t[m]"
            //          << endl;

            //     cout << "Base Euler roll, pitch, yaw:  \t";
            //     Eigen::Vector3d rad = solution.base_angular_->GetPoint(t).p();
            //     cout << (rad / M_PI * 180).transpose() << "\t[deg]" << endl;

            //     cout << "Foot position x,y,z:          \t";
            //     cout << solution.ee_motion_.at(0)->GetPoint(t).p().transpose() << "\t[m]"
            //          << endl;

            //     cout << "Contact force x,y,z:          \t";
            //     cout << solution.ee_force_.at(0)->GetPoint(t).p().transpose() << "\t[N]"
            //          << endl;

            //     bool contact = solution.phase_durations_.at(0)->IsContactPhase(t);
            //     std::string foot_in_contact = contact ? "yes" : "no";
            //     cout << "Foot in contact:              \t" + foot_in_contact << endl;

            //     cout << endl;

            //     t += 0.2;
            // }
        }

        return {-f, -grad, c, C, g, G, cv};
    }
};
using Towr = TowrFit<double>;

struct ParamsPSO {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = Towr::dim;
    static constexpr unsigned int pop_size = 1;
    static constexpr unsigned int num_neighbors = 1;
    static constexpr unsigned int num_neighborhoods = std::floor(pop_size / static_cast<double>(num_neighbors));
    static constexpr double max_value = Towr::max_value;
    static constexpr double min_value = Towr::min_value;
    static constexpr double max_vel = 100.;
    static constexpr double min_vel = -100.;

    // Constraints
    static constexpr unsigned int neq_dim = Towr::neq_dim;
    static constexpr unsigned int nin_dim = Towr::nin_dim;

    static constexpr double chi = 0.729;
    static constexpr double c1 = 2.05;
    static constexpr double c2 = 2.05;
    static constexpr double u = 0.5;

    static constexpr bool noisy_velocity = false;
    static constexpr double mu_noise = 0.;
    static constexpr double sigma_noise = 0.0001;

    static constexpr double qp_alpha = 1.;
    static constexpr double qp_cr = 1.;
    static constexpr double qp_weight = 1.;
    static constexpr double epsilon_comp = 1e-4;
};

int main()
{
    Towr s;
    algevo::algo::ParticleSwarmOptimizationGrad<ParamsPSO, Towr> pso;

    algevo::tools::rgen_gauss_t rgen(0., 0.001);
    // Custom Initialization
    for (unsigned int k = 0; k < ParamsPSO::pop_size; k++) {
        pso.population().row(k) = s.nlp.GetVariableValues();
        if (k > 0)
            pso.population().row(k).array() += rgen.rand();
    }

    // s.eval_all(pso.population().row(0), 3);
    // exit(1);

    for (unsigned int i = 0; i < 500; i++) {
        pso.step();
        // std::cout << i << ": " << pso.best_value() << std::endl;
        std::cout << pso.nfe() << " ";
        s.eval_all(pso.best(), 1);
        std::cin.get();
    }
    std::cout << pso.nfe() << " ";
    s.eval_all(pso.best(), 1);
    std::cout << "Best: " << pso.best() << std::endl;
    return 0;
}
