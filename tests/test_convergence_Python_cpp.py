from linker_cpp import discrete_models
from src.pricers import OptionSpec, crr_price, jr_price, tian93_price

N = 2000

def test_CRR_call_EU_Python_CPP():
    market = discrete_models.MarketData(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
    )

    config = discrete_models.TreeConfig(
        steps=N,
    )

    call = discrete_models.PayoffCall(100);
    option = discrete_models.VanillaOption(call, 1, discrete_models.ExerciseStyle.European)

    cpp_crr_price = discrete_models.CRR_cpp_price(
        option,
        market,
        config,
    )

    spec_crr_call = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Call",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.0,
    )
    print(abs(cpp_crr_price - crr_price(spec_crr_call)))
    assert abs(cpp_crr_price - crr_price(spec_crr_call)) < 1e-6

def test_CRR_put_US_Python_CPP():
    market = discrete_models.MarketData(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
    )

    config = discrete_models.TreeConfig(
        steps=N,
    )

    put = discrete_models.PayoffPut(100);
    option = discrete_models.VanillaOption(put, 1, discrete_models.ExerciseStyle.American)

    cpp_crr_price = discrete_models.CRR_cpp_price(
        option,
        market,
        config,
    )

    spec_crr_put = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Put",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.0,
    )
    print(abs(cpp_crr_price - crr_price(spec_crr_put)))
    assert abs(cpp_crr_price - crr_price(spec_crr_put)) < 1e-6

def test_JR_call_EU_Python_CPP():
    market = discrete_models.MarketData(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
    )

    config = discrete_models.TreeConfig(
        steps=N,
    )

    call = discrete_models.PayoffCall(100);
    option = discrete_models.VanillaOption(call, 1, discrete_models.ExerciseStyle.European)

    cpp_jr_price = discrete_models.JR_cpp_price(
        option,
        market,
        config,
    )

    spec_jr_call = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Call",
        exercise_style="European",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )
    print(abs(cpp_jr_price - jr_price(spec_jr_call)))
    assert abs(cpp_jr_price - jr_price(spec_jr_call)) < 1e-6

def test_JR_put_US_Python_CPP():
    market = discrete_models.MarketData(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
    )

    config = discrete_models.TreeConfig(
        steps=N,
    )

    put = discrete_models.PayoffPut(100);
    option = discrete_models.VanillaOption(put, 1, discrete_models.ExerciseStyle.American)

    cpp_jr_price = discrete_models.JR_cpp_price(
        option,
        market,
        config,
    )

    spec_jr_put = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Put",
        exercise_style="American",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )
    print(abs(cpp_jr_price - jr_price(spec_jr_put)))
    assert abs(cpp_jr_price - jr_price(spec_jr_put)) < 1e-6

def test_TIAN93_call_EU_Python_CPP():
    market = discrete_models.MarketData(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
    )

    config = discrete_models.TreeConfig(
         steps=N,
    )

    call = discrete_models.PayoffCall(100);
    option = discrete_models.VanillaOption(call, 1, discrete_models.ExerciseStyle.European)

    cpp_tian93_price = discrete_models.TIAN93_cpp_price(
        option,
        market,
        config,
    )

    spec_tian93_call = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Call",
        exercise_style="European",
        model="Tian-1993",
        dividend_yield=0.0,
        )
    print(abs(cpp_tian93_price - tian93_price(spec_tian93_call)))
    assert abs(cpp_tian93_price - tian93_price(spec_tian93_call)) < 1e-6

def test_TIAN93_put_US_Python_CPP():
        market = discrete_models.MarketData(
            spot=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.20,
        )

        config = discrete_models.TreeConfig(
            steps=N,
        )

        put = discrete_models.PayoffPut(100);
        option = discrete_models.VanillaOption(put, 1, discrete_models.ExerciseStyle.American)

        cpp_tian93_price = discrete_models.TIAN93_cpp_price(
            option,
            market,
            config,
        )

        spec_tian93_put = OptionSpec(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            steps=N,
            option_type="Put",
            exercise_style="American",
            model="Tian-1993",
            dividend_yield=0.0,
        )
        print(abs(cpp_tian93_price - tian93_price(spec_tian93_put)))
        assert abs(cpp_tian93_price - tian93_price(spec_tian93_put)) < 1e-6