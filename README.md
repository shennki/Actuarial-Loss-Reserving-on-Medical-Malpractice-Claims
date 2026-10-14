## Loss Reserving And Backtesting Validation On Medical Malpractice Claims

An actuarial reserving algorithm and validation framework built for long-term Medical Malpractice (MedMal) insurance lines. This project implements a chain-ladder model for loss reserving alongside a backtesting framework to test model stability and manage projection volatility in recent accident years.

##

The project is divided into two operational scripts:

1. **Loss Reserving (`chain_ladder_model.py`)**: Processes the complete historical loss triangle to project ultimate losses and calculate current required IBNR (Incurred But Not Reported) reserves.
2. **Backtesting Validation (`backtest_validation.py`)**: Implements diagonal stair-step truncation to simulate past evaluation dates to evaluate model performance.

##

Methodology:

* **Chain Ladder Method**: Computes age-to-age development factors (link ratios) and cumulative age-to-ultimate factors from historical paid loss triangles.
* **Bornhuetter-Ferguson (BF) Method**: Mitigates severe under-projections in recent, sparse accident years by blending actual paid losses with an *a priori* expected loss baseline weighted by the percentage of unpaid losses.

##

Running the Code:
1. **Install dependencies**:
pip install pandas numpy
2. Place **(`medmal_pos_98-07.csv`)** in the root directory.
3. **Run production reserving**:
python chain_ladder_model.py
4. **Run backtesting validation**:
python backtest_validation.py

##

Limitations and Future Improvements:
1. The dataset is constrained to 10 development lags, requiring the model to assume zero claim activity past year 10 (a 1.0 tail factor). In real-world MedMal insurance, liabilities frequently extend beyond 10 years. A production extension would incorporate tail-factor curve fitting, such as an inverse power curve.
2. Multi-Company Automation: The current scripts process a single company code; future iterations can scale via a loop to process all entities in the dataset for portfolio-wide aggregation.