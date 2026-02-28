// Gurobi C++ API shim — delegates to gurobipy via pybind11.
// Placed on the include path BEFORE system/vendor headers so
// #include "gurobi_c++.h" resolves here instead of the real Gurobi SDK.
//
// Implements only the subset used by HeiCut (lib/solvers/ilp.h/cpp):
//   GRBException, GRBEnv, GRBVar, GRBLinExpr, GRBModel
//   Constants: GRB_BINARY, GRB_CONTINUOUS, GRB_MINIMIZE, GRB_OPTIMAL, etc.

#ifndef GUROBI_CPP_SHIM_H
#define GUROBI_CPP_SHIM_H

#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants  (values match the real Gurobi SDK)
// ---------------------------------------------------------------------------
// Variable types
constexpr char GRB_BINARY     = 'B';
constexpr char GRB_CONTINUOUS = 'C';
constexpr char GRB_INTEGER    = 'I';

// Objective sense
constexpr int GRB_MINIMIZE = 1;
constexpr int GRB_MAXIMIZE = -1;

// Optimization status
constexpr int GRB_OPTIMAL    = 2;
constexpr int GRB_INFEASIBLE = 3;
constexpr int GRB_TIME_LIMIT = 9;

// Attribute tags — we use sentinel ints; the shim interprets them internally.
constexpr int GRB_IntAttr_Status   = 1001;
constexpr int GRB_DoubleAttr_X     = 2001;
constexpr int GRB_DoubleAttr_Start = 2002;

// String/Double/Int param tags
constexpr int GRB_StringAttr_ModelName      = 3001;
constexpr int GRB_DoubleParam_MIPGap        = 4001;
constexpr int GRB_DoubleParam_IntFeasTol    = 4002;
constexpr int GRB_DoubleParam_FeasibilityTol= 4003;
constexpr int GRB_DoubleParam_TimeLimit     = 4004;
constexpr int GRB_IntParam_LogToConsole     = 5001;
constexpr int GRB_IntParam_Seed             = 5002;
constexpr int GRB_IntParam_Threads          = 5003;

// ---------------------------------------------------------------------------
// GRBException
// ---------------------------------------------------------------------------
class GRBException : public std::exception {
    std::string msg_;
    int code_;
public:
    GRBException() : msg_("Gurobi error"), code_(0) {}
    GRBException(const std::string& m, int c = 0) : msg_(m), code_(c) {}
    const char* what() const noexcept override { return msg_.c_str(); }
    int getErrorCode() const { return code_; }
    std::string getMessage() const { return msg_; }
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
class GRBModel;

// ---------------------------------------------------------------------------
// GRBVar — lightweight handle referencing a variable index in a GRBModel
// ---------------------------------------------------------------------------
class GRBVar {
    friend class GRBModel;
    friend class GRBLinExpr;
    GRBModel* model_ = nullptr;
    int idx_ = -1;
public:
    GRBVar() = default;
    GRBVar(GRBModel* m, int i) : model_(m), idx_(i) {}

    // get(GRB_DoubleAttr_X) → solution value
    // get(GRB_DoubleAttr_Start) → start hint (ignored)
    double get(int attr) const;

    // set(GRB_DoubleAttr_Start, val) → store start hint
    void set(int attr, double val);
};

// ---------------------------------------------------------------------------
// GRBLinExpr — stores  constant + sum(coeff_i * var_i)
// ---------------------------------------------------------------------------
class GRBLinExpr {
    friend class GRBModel;
public:
    double constant_ = 0.0;
    std::vector<std::pair<double, int>> terms_; // (coeff, var_idx)
    GRBLinExpr() = default;
    GRBLinExpr(double c) : constant_(c) {}

    GRBLinExpr& operator+=(const GRBVar& v) {
        terms_.emplace_back(1.0, v.idx_);
        return *this;
    }
    GRBLinExpr& operator+=(const GRBLinExpr& o) {
        constant_ += o.constant_;
        terms_.insert(terms_.end(), o.terms_.begin(), o.terms_.end());
        return *this;
    }

    // Implicit conversion from GRBVar
    GRBLinExpr(const GRBVar& v) {
        terms_.emplace_back(1.0, v.idx_);
    }
};

// ---------------------------------------------------------------------------
// Free-standing operators for GRBVar arithmetic
// ---------------------------------------------------------------------------

// scalar * GRBVar → GRBLinExpr
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline GRBLinExpr operator*(T c, const GRBVar& v) {
    GRBLinExpr e;
    e += v;
    // Manually scale the coefficient
    if (!e.terms_.empty()) e.terms_.back().first = static_cast<double>(c);
    return e;
}

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline GRBLinExpr operator*(const GRBVar& v, T c) {
    return c * v;
}

// GRBVar - GRBVar → GRBLinExpr
inline GRBLinExpr operator-(const GRBVar& a, const GRBVar& b) {
    GRBLinExpr e;
    e += a;
    GRBLinExpr neg;
    neg += b;
    neg.terms_.back().first = -1.0;
    e += neg;
    return e;
}

// GRBLinExpr - GRBLinExpr → GRBLinExpr  (for maximums[e] - minimums[e])
inline GRBLinExpr operator-(const GRBLinExpr& a, const GRBLinExpr& b) {
    GRBLinExpr result = a;
    result.constant_ -= b.constant_;
    for (auto& [coeff, idx] : b.terms_)
        result.terms_.emplace_back(-coeff, idx);
    return result;
}

// ---------------------------------------------------------------------------
// GRBEnv — in the real SDK this holds license/config; we store nothing.
// ---------------------------------------------------------------------------
class GRBEnv {
public:
    GRBEnv() = default;
    explicit GRBEnv(bool /*start*/) {}
    void start() {}
    void set(const std::string& /*param*/, const std::string& /*value*/) {}
};

// ---------------------------------------------------------------------------
// Constraint sense helpers
// ---------------------------------------------------------------------------
enum GRBConstrSense { GRB_GE, GRB_LE, GRB_EQ };

struct GRBTempConstr {
    GRBLinExpr lhs;
    GRBConstrSense sense;
    GRBLinExpr rhs;
};

// operator>= between GRBLinExpr and GRBLinExpr / int / GRBVar
inline GRBTempConstr operator>=(const GRBLinExpr& lhs, const GRBLinExpr& rhs) {
    return {lhs, GRB_GE, rhs};
}
inline GRBTempConstr operator>=(const GRBLinExpr& lhs, int rhs) {
    return {lhs, GRB_GE, GRBLinExpr(static_cast<double>(rhs))};
}
inline GRBTempConstr operator>=(const GRBLinExpr& lhs, unsigned int rhs) {
    return {lhs, GRB_GE, GRBLinExpr(static_cast<double>(rhs))};
}
inline GRBTempConstr operator>=(const GRBVar& lhs, const GRBVar& rhs) {
    GRBLinExpr l; l += lhs;
    GRBLinExpr r; r += rhs;
    return {l, GRB_GE, r};
}

inline GRBTempConstr operator<=(const GRBLinExpr& lhs, const GRBLinExpr& rhs) {
    return {lhs, GRB_LE, rhs};
}
inline GRBTempConstr operator<=(const GRBLinExpr& lhs, int rhs) {
    return {lhs, GRB_LE, GRBLinExpr(static_cast<double>(rhs))};
}
inline GRBTempConstr operator<=(const GRBLinExpr& lhs, unsigned int rhs) {
    return {lhs, GRB_LE, GRBLinExpr(static_cast<double>(rhs))};
}
inline GRBTempConstr operator<=(const GRBVar& lhs, const GRBVar& rhs) {
    GRBLinExpr l; l += lhs;
    GRBLinExpr r; r += rhs;
    return {l, GRB_LE, r};
}

// ---------------------------------------------------------------------------
// Internal storage types for GRBModel
// ---------------------------------------------------------------------------
struct VarSpec {
    double lb, ub, obj;
    char vtype;
    double startVal;
    double solVal;
};

struct ConstrSpec {
    GRBLinExpr lhs;
    GRBConstrSense sense;
    GRBLinExpr rhs;
    std::string name;
};

// ---------------------------------------------------------------------------
// GRBModel — accumulates variables, constraints, objective in C++,
//             then at optimize() serializes to gurobipy and solves.
// ---------------------------------------------------------------------------
class GRBModel {
    friend class GRBVar;

    std::vector<VarSpec> vars_;
    std::vector<ConstrSpec> constrs_;
    GRBLinExpr objective_;
    int objSense_ = GRB_MINIMIZE;
    int status_ = 0;

    // Model parameters
    std::string modelName_;
    double mipGap_ = 1e-4;
    double intFeasTol_ = 1e-5;
    double feasibilityTol_ = 1e-6;
    double timeLimit_ = 1e100;
    int logToConsole_ = 1;
    int seed_ = 0;
    int threads_ = 0;

public:
    GRBModel() = default;
    explicit GRBModel(const GRBEnv& /*env*/) {}

    // --- addVar ---
    GRBVar addVar(double lb, double ub, double obj, char vtype) {
        int idx = static_cast<int>(vars_.size());
        vars_.push_back({lb, ub, obj, vtype, 0.0, 0.0});
        return GRBVar(this, idx);
    }

    // --- addConstr ---
    void addConstr(const GRBTempConstr& c, const std::string& name = "") {
        constrs_.push_back({c.lhs, c.sense, c.rhs, name});
    }

    // --- setObjective ---
    void setObjective(const GRBLinExpr& expr, int sense = GRB_MINIMIZE) {
        objective_ = expr;
        objSense_ = sense;
    }

    // --- set (string attr) ---
    void set(int attr, const std::string& val) {
        if (attr == GRB_StringAttr_ModelName) modelName_ = val;
    }
    void set(int attr, const char* val) { set(attr, std::string(val)); }

    // --- set (double param) ---
    void set(int attr, double val) {
        switch (attr) {
            case GRB_DoubleParam_MIPGap:         mipGap_ = val; break;
            case GRB_DoubleParam_IntFeasTol:     intFeasTol_ = val; break;
            case GRB_DoubleParam_FeasibilityTol: feasibilityTol_ = val; break;
            case GRB_DoubleParam_TimeLimit:       timeLimit_ = val; break;
            default: break;
        }
    }

    // --- set (int param) ---
    void set(int attr, int val) {
        switch (attr) {
            case GRB_IntParam_LogToConsole: logToConsole_ = val; break;
            case GRB_IntParam_Seed:         seed_ = val; break;
            case GRB_IntParam_Threads:      threads_ = val; break;
            default: break;
        }
    }
    void set(int attr, size_t val) { set(attr, static_cast<int>(val)); }

    // --- get (int attr) ---
    int get(int attr) const {
        if (attr == GRB_IntAttr_Status) return status_;
        return 0;
    }

    // --- optimize: the core method that calls gurobipy ---
    void optimize() {
        py::gil_scoped_acquire gil;
        try {
            py::module_ gp = py::module_::import("gurobipy");
            py::object Model = gp.attr("Model")(modelName_);
            py::object GRB_obj = gp.attr("GRB");

            // Set parameters
            Model.attr("setParam")("MIPGap", mipGap_);
            Model.attr("setParam")("IntFeasTol", intFeasTol_);
            Model.attr("setParam")("FeasibilityTol", feasibilityTol_);
            Model.attr("setParam")("TimeLimit", timeLimit_);
            Model.attr("setParam")("LogToConsole", logToConsole_);
            Model.attr("setParam")("Seed", seed_);
            if (threads_ > 0)
                Model.attr("setParam")("Threads", threads_);

            // Variable type mapping
            py::object BINARY_type = GRB_obj.attr("BINARY");
            py::object CONTINUOUS_type = GRB_obj.attr("CONTINUOUS");
            py::object INTEGER_type = GRB_obj.attr("INTEGER");

            // Add variables
            std::vector<py::object> pyVars(vars_.size());
            for (size_t i = 0; i < vars_.size(); ++i) {
                py::object vt;
                switch (vars_[i].vtype) {
                    case GRB_BINARY:     vt = BINARY_type; break;
                    case GRB_CONTINUOUS: vt = CONTINUOUS_type; break;
                    case GRB_INTEGER:    vt = INTEGER_type; break;
                    default:             vt = BINARY_type; break;
                }
                pyVars[i] = Model.attr("addVar")(
                    py::arg("lb") = vars_[i].lb,
                    py::arg("ub") = vars_[i].ub,
                    py::arg("obj") = vars_[i].obj,
                    py::arg("vtype") = vt
                );
                if (vars_[i].startVal != 0.0) {
                    pyVars[i].attr("Start") = vars_[i].startVal;
                }
            }

            // Helper: convert GRBLinExpr to gurobipy LinExpr
            auto to_py_expr = [&](const GRBLinExpr& e) -> py::object {
                py::object LinExpr = gp.attr("LinExpr")();
                if (e.constant_ != 0.0)
                    LinExpr.attr("addConstant")(e.constant_);
                for (auto& [coeff, idx] : e.terms_) {
                    LinExpr.attr("addTerms")(
                        py::make_tuple(coeff),
                        py::make_tuple(pyVars[idx])
                    );
                }
                return LinExpr;
            };

            // Add constraints
            for (auto& c : constrs_) {
                py::object lhs = to_py_expr(c.lhs);
                py::object rhs = to_py_expr(c.rhs);
                char sense_char;
                switch (c.sense) {
                    case GRB_GE: sense_char = '>'; break;
                    case GRB_LE: sense_char = '<'; break;
                    case GRB_EQ: sense_char = '='; break;
                }
                Model.attr("addLConstr")(lhs, sense_char, rhs, py::arg("name") = c.name);
            }

            // Set objective
            py::object objExpr = to_py_expr(objective_);
            int pySense = (objSense_ == GRB_MINIMIZE) ? 1 : -1;
            Model.attr("setObjective")(objExpr, pySense);

            // Optimize
            Model.attr("optimize")();

            // Read status
            int pyStatus = Model.attr("Status").cast<int>();
            status_ = pyStatus; // gurobipy uses same status codes

            // Read solution values
            if (pyStatus == GRB_OPTIMAL || pyStatus == GRB_TIME_LIMIT) {
                for (size_t i = 0; i < vars_.size(); ++i) {
                    try {
                        vars_[i].solVal = pyVars[i].attr("X").cast<double>();
                    } catch (...) {
                        vars_[i].solVal = 0.0;
                    }
                }
            }
        } catch (py::error_already_set& e) {
            status_ = 0;
            throw GRBException(e.what(), -1);
        }
    }
};

// ---------------------------------------------------------------------------
// GRBVar method implementations (need GRBModel to be complete)
// ---------------------------------------------------------------------------
inline double GRBVar::get(int attr) const {
    if (!model_ || idx_ < 0) return 0.0;
    if (attr == GRB_DoubleAttr_X)
        return model_->vars_[idx_].solVal;
    if (attr == GRB_DoubleAttr_Start)
        return model_->vars_[idx_].startVal;
    return 0.0;
}

inline void GRBVar::set(int attr, double val) {
    if (!model_ || idx_ < 0) return;
    if (attr == GRB_DoubleAttr_Start)
        model_->vars_[idx_].startVal = val;
}

#endif // GUROBI_CPP_SHIM_H
