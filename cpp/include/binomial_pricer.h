#pragma once
#include<memory>

namespace discrete_models {
    enum class ExerciseStyle {
        European,
        American
    };

    class Payoff {          // ABC Class
    public:
        Payoff() = default;
        Payoff(const Payoff&) = default;
        Payoff& operator=(const Payoff&) = default;
        Payoff(Payoff&&) noexcept = default;
        Payoff& operator=(Payoff&&) noexcept = default;

        virtual ~Payoff() = default;

        [[nodiscard]]
        virtual double strike() const noexcept = 0;

        [[nodiscard]]
        virtual double operator()(double spot) const = 0;

        [[nodiscard]]
        virtual std::unique_ptr<Payoff> clone() const = 0;
    };

    class PayoffCall final : public Payoff {
    public:
        explicit PayoffCall(double strike);

        [[nodiscard]]
        double strike() const noexcept override {
            return strike_;
        }

        [[nodiscard]]
        double operator()(double spot) const override;

        [[nodiscard]]
        std::unique_ptr<Payoff> clone() const override;
    private:
        double strike_;
    };

    class PayoffPut final : public Payoff {
    public:
        explicit PayoffPut(double strike);

        [[nodiscard]]
        double strike() const noexcept override {
            return strike_;
        }

        [[nodiscard]]
        double operator()(double spot) const override;

        [[nodiscard]]
        std::unique_ptr<Payoff> clone() const override;
    private:
        double strike_;
    };

    class VanillaOption {
        public:
        VanillaOption(const Payoff& payoff,  double maturity,  ExerciseStyle exercise_style);

        [[nodiscard]]
        double strike() const noexcept;

        friend void swap(VanillaOption& lhs, VanillaOption& rhs) noexcept {
            using std::swap;
            swap(lhs.payoff_, rhs.payoff_);
            swap(lhs.maturity_, rhs.maturity_);
            swap(lhs.exercise_style_, rhs.exercise_style_);
        }

        // Deep copy management
        VanillaOption(const VanillaOption& other);
        VanillaOption& operator=(const VanillaOption& other);

        // Ownership transfer
        VanillaOption(VanillaOption&& other) noexcept = default;
        VanillaOption& operator=(VanillaOption&& other) noexcept = default;

        ~VanillaOption() = default;

        [[nodiscard]]
        double payoff(double spot) const;

        [[nodiscard]]
        double maturity() const noexcept;

        [[nodiscard]]
        ExerciseStyle exercise_style() const noexcept;

    private:
        std::unique_ptr<Payoff> payoff_;
        double maturity_;
        ExerciseStyle exercise_style_;
    };

    struct MarketData {
        double spot;
        double rate;
        double dividend_yield;
        double volatility;
    };

    struct TreeConfig {
        int steps;
    };

    struct TreeParameters {
        double u;
        double d;
        double p;
    };

    struct BinomialResult {
        double price;
        double root_continuation;
        double root_intrinsic;
    };

    [[nodiscard]]
    double CRR_price(const VanillaOption& option, const MarketData& market, const TreeConfig& config);

    [[nodiscard]]
    double JR_price(const VanillaOption& option, const MarketData& market, const TreeConfig& config);

    [[nodiscard]]
    double TIAN93_price(const VanillaOption& option, const MarketData& market, const TreeConfig& config);

    [[nodiscard]]
    double LR_price(const VanillaOption& option, const MarketData& market, const TreeConfig& config);

    [[nodiscard]]
    double LR_root_exercise_gap(const VanillaOption& option, const MarketData& market, const TreeConfig& config);

    [[nodiscard]]
    double TIAN99_price(const VanillaOption& option, const MarketData& market, const TreeConfig& config);

    [[nodiscard]]
    double binomial_price(const VanillaOption& option, const MarketData& market, const TreeConfig& config,
        const TreeParameters& TreeParams);

}  // namespace discrete_models
