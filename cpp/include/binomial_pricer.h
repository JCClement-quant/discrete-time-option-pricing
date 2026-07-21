#pragma once
#include<memory>

namespace discrete_models {
    enum class ExerciseStyle {
        European,
        American
    };

    class Payoff {          // ABC Class
    public:
        virtual ~Payoff() = default;

        [[nodiscard]]
        virtual double operator()(double spot) const = 0;

        [[nodiscard]]
        virtual std::unique_ptr<Payoff> clone() const = 0;
    };

    class PayoffCall final : public Payoff {
    public:
        explicit PayoffCall(double strike);

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
        double operator()(double spot) const override;

        [[nodiscard]]
        std::unique_ptr<Payoff> clone() const override;
    private:
        double strike_;
    };

    class VanillaOption {
        public:
        VanillaOption(Payoff& payoff, double maturity, ExerciseStyle exercise_style);

        // Deep copy management
        VanillaOption(const VanillaOption& other);
        VanillaOption& operator=(const VanillaOption& other);

        // Ownership transfer
        VanillaOption(VanillaOption&& other) noexcept = default;
        VanillaOption& operator=(VanillaOption&& other) noexcept = default;

        ~VanillaOption() = default;

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

}  // namespace discrete_models
