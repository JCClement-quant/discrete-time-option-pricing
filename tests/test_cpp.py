import discrete_models

def test_payoff_call():
    # C++ PayoffCall instance
    call = discrete_models.PayoffCall(100.0)

    # Payoff check ITM
    assert call(115.0) == 15.0

    # Payoff check OTM
    assert call(85.0) == 0.0

def test_option_properties():
    call = discrete_models.PayoffCall(100.0)
    option = discrete_models.VanillaOption(
        call,
        1.5,
        discrete_models.ExerciseStyle.European
    )

    assert option.maturity == 1.5
    assert option.exercise_style == discrete_models.ExerciseStyle.European