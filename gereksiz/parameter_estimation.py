def parameter_estimation(data):

    import numpy as np
    from scipy.optimize import minimize

    # ============================================================
    # 1. DATA
    # ============================================================

    if data is None:
        raise ValueError(
            "Market data bulunamadı. "
            "data.py'den gelen DataFrame'i "
            "parameter_estimation(data) şeklinde gönderin."
        )

    if "Close" not in data.columns:
        raise ValueError(
            "Veri içerisinde 'Close' sütunu bulunamadı."
        )

    data = data[["Close"]].copy()

    data["Close"] = data["Close"].astype(float)

    data = data.dropna(
        subset=["Close"]
    )

    # ============================================================
    # 2. LOG RETURNS
    # ============================================================

    data["Log_Return"] = np.log(
        data["Close"] /
        data["Close"].shift(1)
    )

    data = data.dropna(
        subset=["Log_Return"]
    )

    returns = data["Log_Return"].to_numpy(
        dtype=float
    )

    # ============================================================
    # 3. TIME PARAMETERS
    # ============================================================

    TRADING_DAYS = 252

    dt = 1 / TRADING_DAYS

    n = len(returns)

    if n < 2:
        raise ValueError(
            "GBM parameter estimation için yeterli "
            "gözlem bulunamadı."
        )

    # ============================================================
    # 4. ANALYTICAL GBM MLE
    # ============================================================
    #
    # GBM:
    #
    # dS_t = mu*S_t*dt + sigma*S_t*dW_t
    #
    # Log-return:
    #
    # r_t ~ N(
    #     (mu - 0.5*sigma^2)*dt,
    #     sigma^2*dt
    # )
    #
    # Analytical MLE:
    #
    # sigma^2_hat =
    #     1/(n*dt) * sum((r_t - mean_r)^2)
    #
    # mu_hat =
    #     mean_r/dt + 0.5*sigma^2_hat
    #
    # ============================================================

    mean_return = np.mean(returns)

    sigma_squared_analytical = (
        np.sum(
            (returns - mean_return) ** 2
        )
        / (n * dt)
    )

    sigma_analytical = np.sqrt(
        sigma_squared_analytical
    )

    mu_analytical = (
        mean_return / dt
        + 0.5 * sigma_squared_analytical
    )

    # ============================================================
    # 5. GBM NEGATIVE LOG-LIKELIHOOD
    # ============================================================
    #
    # Burada sigma'yı doğrudan optimize etmek yerine:
    #
    #     eta = log(sigma)
    #
    # kullanacağız.
    #
    # Böylece:
    #
    #     sigma = exp(eta)
    #
    # olduğundan sigma her zaman pozitiftir.
    #
    # Bu yaklaşım numerical optimization açısından
    # daha stabildir.
    #
    # ============================================================

    def negative_log_likelihood(
        params,
        returns,
        dt
    ):

        mu, log_sigma = params

        sigma = np.exp(log_sigma)

        # GBM log-return mean
        mean = (
            mu
            - 0.5 * sigma**2
        ) * dt

        # GBM log-return variance
        variance = (
            sigma**2 * dt
        )

        residuals = (
            returns - mean
        )

        n = len(returns)

        nll = (
            0.5
            * n
            * np.log(
                2 * np.pi * variance
            )
            +
            np.sum(
                residuals**2
            )
            /
            (2 * variance)
        )

        return nll

    # ============================================================
    # 6. ANALYTICAL GRADIENT OF NLL
    # ============================================================
    #
    # Numerical optimizer'ın gradient'i finite difference ile
    # tahmin etmesi yerine gradient'i kendimiz veriyoruz.
    #
    # Bu, özellikle büyük örneklem ve hassas likelihood
    # problemlerinde daha stabil çalışır.
    #
    # ============================================================

    def negative_log_likelihood_gradient(
        params,
        returns,
        dt
    ):

        mu, log_sigma = params

        sigma = np.exp(log_sigma)

        n = len(returns)

        # Mean of GBM log-return
        mean = (
            mu
            - 0.5 * sigma**2
        ) * dt

        # Variance
        variance = (
            sigma**2 * dt
        )

        # Residuals
        residuals = (
            returns - mean
        )

        sum_residuals = np.sum(
            residuals
        )

        sum_squared_residuals = np.sum(
            residuals**2
        )

        # --------------------------------------------------------
        # Derivative with respect to mu
        # --------------------------------------------------------

        d_nll_d_mu = (
            -dt
            * sum_residuals
            / variance
        )

        # --------------------------------------------------------
        # Derivative with respect to sigma
        # --------------------------------------------------------

        d_nll_d_sigma = (
            n / sigma
            +
            (
                sigma
                * dt
                * sum_residuals
                / variance
            )
            -
            (
                sum_squared_residuals
                * sigma
                * dt
                / variance**2
            )
        )

        # --------------------------------------------------------
        # Since:
        #
        # sigma = exp(log_sigma)
        #
        # d sigma / d log_sigma = sigma
        #
        # therefore:
        #
        # dNLL/dlog_sigma =
        #     dNLL/dsigma * sigma
        # --------------------------------------------------------

        d_nll_d_log_sigma = (
            d_nll_d_sigma
            * sigma
        )

        return np.array(
            [
                d_nll_d_mu,
                d_nll_d_log_sigma
            ]
        )

    # ============================================================
    # 7. NUMERICAL GBM MLE
    # ============================================================
    #
    # Instead of:
    #
    #     [mu, sigma]
    #
    # we optimize:
    #
    #     [mu, log(sigma)]
    #
    # ============================================================

    initial_guess = np.array(
        [
            mu_analytical,
            np.log(sigma_analytical)
        ],
        dtype=float
    )

    result = minimize(
        negative_log_likelihood,
        x0=initial_guess,
        args=(returns, dt),
        jac=negative_log_likelihood_gradient,
        method="BFGS",
        options={
            "gtol": 1e-10,
            "maxiter": 1000
        }
    )

    # ============================================================
    # 8. NUMERICAL RESULTS
    # ============================================================

    mu_numerical = result.x[0]

    sigma_numerical = np.exp(
        result.x[1]
    )

    # ============================================================
    # 9. NUMERICAL OPTIMIZATION CHECK
    # ============================================================

    # BFGS bazen optimum noktaya çok yakın olduğu halde
    # precision nedeniyle success=False döndürebilir.
    #
    # Bu nedenle sadece result.success'e bakmıyoruz.
    # Analytical ve numerical likelihood sonuçlarını da
    # karşılaştırıyoruz.
    #
    # ============================================================

    analytical_nll = negative_log_likelihood(
        [
            mu_analytical,
            np.log(sigma_analytical)
        ],
        returns,
        dt
    )

    numerical_nll = negative_log_likelihood(
        [
            mu_numerical,
            np.log(sigma_numerical)
        ],
        returns,
        dt
    )

    # ============================================================
    # 10. LIKELIHOOD VALUES
    # ============================================================

    analytical_log_likelihood = (
        -analytical_nll
    )

    numerical_log_likelihood = (
        -numerical_nll
    )

    # ============================================================
    # 11. SAMPLE ESTIMATES
    # ============================================================

    sample_sigma_daily = np.std(
        returns,
        ddof=1
    )

    sample_sigma_annual = (
        sample_sigma_daily
        * np.sqrt(TRADING_DAYS)
    )

    sample_mean_annual = (
        mean_return
        * TRADING_DAYS
    )

    # ============================================================
    # 12. OPTIMIZATION DIAGNOSTICS
    # ============================================================

    drift_difference = abs(
        mu_analytical
        - mu_numerical
    )

    volatility_difference = abs(
        sigma_analytical
        - sigma_numerical
    )

    nll_difference = abs(
        analytical_nll
        - numerical_nll
    )

    gradient_at_solution = (
        negative_log_likelihood_gradient(
            result.x,
            returns,
            dt
        )
    )

    gradient_norm = np.linalg.norm(
        gradient_at_solution
    )

    # Analytical and numerical MLE should be
    # essentially identical.
    #
    # The tolerance is intentionally numerical,
    # not statistical.

    numerical_matches_analytical = (
        np.isclose(
            mu_analytical,
            mu_numerical,
            rtol=1e-6,
            atol=1e-10
        )
        and
        np.isclose(
            sigma_analytical,
            sigma_numerical,
            rtol=1e-6,
            atol=1e-10
        )
        and
        np.isclose(
            analytical_nll,
            numerical_nll,
            rtol=1e-10,
            atol=1e-8
        )
    )

    # ============================================================
    # 13. RESULTS
    # ============================================================

    print()
    print("=" * 70)
    print("GBM PARAMETER ESTIMATION")
    print("=" * 70)

    print()
    print("DATA")
    print("-" * 70)

    print(
        f"Observations : {n}"
    )

    print(
        f"Start        : "
        f"{data.index.min().date()}"
    )

    print(
        f"End          : "
        f"{data.index.max().date()}"
    )

    print(
        f"Trading days : {TRADING_DAYS}"
    )

    print(
        f"dt           : {dt:.8f}"
    )

    # ============================================================
    # ANALYTICAL MLE
    # ============================================================

    print()
    print("=" * 70)
    print("ANALYTICAL GBM MLE")
    print("=" * 70)

    print(
        f"Annual drift (mu)        : "
        f"{mu_analytical:.8%}"
    )

    print(
        f"Annual volatility (sigma): "
        f"{sigma_analytical:.8%}"
    )

    print(
        f"Log-likelihood            : "
        f"{analytical_log_likelihood:.6f}"
    )

    # ============================================================
    # NUMERICAL MLE
    # ============================================================

    print()
    print("=" * 70)
    print("NUMERICAL GBM MLE")
    print("=" * 70)

    print(
        f"Annual drift (mu)        : "
        f"{mu_numerical:.8%}"
    )

    print(
        f"Annual volatility (sigma): "
        f"{sigma_numerical:.8%}"
    )

    print(
        f"Log-likelihood            : "
        f"{numerical_log_likelihood:.6f}"
    )

    print(
        f"Optimization success     : "
        f"{result.success}"
    )

    print(
        f"Optimizer message        : "
        f"{result.message}"
    )

    print(
        f"Gradient norm            : "
        f"{gradient_norm:.12e}"
    )

    # ============================================================
    # ANALYTICAL vs NUMERICAL
    # ============================================================

    print()
    print("=" * 70)
    print("ANALYTICAL vs NUMERICAL MLE")
    print("=" * 70)

    print(
        f"Drift difference         : "
        f"{drift_difference:.12e}"
    )

    print(
        f"Volatility difference    : "
        f"{volatility_difference:.12e}"
    )

    print(
        f"NLL difference           : "
        f"{nll_difference:.12e}"
    )

    print(
        f"Numerical MLE validated  : "
        f"{numerical_matches_analytical}"
    )

    # ============================================================
    # INTERMEDIATE VALUES
    # ============================================================

    print()
    print("=" * 70)
    print("INTERMEDIATE VALUES")
    print("=" * 70)

    print(
        f"Mean daily log return    : "
        f"{mean_return:.8f}"
    )

    print(
        f"MLE variance             : "
        f"{sigma_squared_analytical:.8f}"
    )

    print(
        f"MLE daily volatility     : "
        f"{sigma_analytical / np.sqrt(TRADING_DAYS):.8f}"
    )

    # ============================================================
    # SAMPLE COMPARISON
    # ============================================================

    print()
    print("=" * 70)
    print("SAMPLE ESTIMATE CHECK")
    print("=" * 70)

    print(
        f"Annualized mean return   : "
        f"{sample_mean_annual:.8%}"
    )

    print(
        f"Annualized volatility    : "
        f"{sample_sigma_annual:.8%}"
    )

    # ============================================================
    # 14. FINAL RETURN
    # ============================================================

    return mu_numerical, sigma_numerical