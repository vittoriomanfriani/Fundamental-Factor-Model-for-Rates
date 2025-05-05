# Fundamental Factor Model for Rates

This repository provides a comprehensive framework for constructing and analyzing yield curves using the **Nelson-Siegel** and **Nelson-Siegel-Svensson** models. It also provides tools to assess bond excess returns after removing the effects of carry and roll-down.

---

## Project Structure

### 1. Models
- **`Nelson_Siegel.py`**: Implements the Nelson-Siegel yield curve model.
- **`Nelson_Siegel_Svensson.py`**: Extends the Nelson-Siegel model to include additional flexibility for the yield curve shape.

### 2. Returns
- **`Carry.py`**: Calculates the carry (return due to accrued interest and coupon payments) for a bond.
- **`RollDown.py`**: Calculates the roll-down return (price change due to bond maturity shifting closer).

### 3. SpotCurve
- **`Spot_Curve_Calculator.py`**: Constructs spot rate curves using bond prices and yields.

### 4. Utils
- **`conversions.py`**: Converts data formats and units used in the analysis.
- **`CrossSectional_Regression.py`**: Performs regressions to explain bond excess returns using yield curve factors.
- **`data_processing.py`**: Prepares data for yield curve fitting and backtesting.
- **`get_spot_rates.py`**: Extracts spot rates from yield data.

---

## Theoretical Background

### 1. Spot Curve Construction
The **spot curve** represents the yields of zero-coupon bonds as a function of their maturity. In this project:
1. **On-the-Run Bonds**: We use market prices of liquid, on-the-run bonds to derive the spot curve.
2. **Bootstrapping**: The spot rates are calculated iteratively, ensuring consistency with bond prices.
3. The **spot curve** is the foundation for fitting parametric models like Nelson-Siegel.

---

### 2. Nelson-Siegel Model
The Nelson-Siegel model is a parametric representation of the yield curve, defined as:
$y(t) = \beta_0 + \beta_1 \frac{1 - e^{-\lambda t}}{\lambda t} + \beta_2 \left(\frac{1 - e^{-\lambda t}}{\lambda t} - e^{-\lambda t}\right)$

- $\beta_0$: Long-term level of yields.
- $\beta_1$: Short-term component.
- $\beta_2$: Medium-term "hump" or curvature.
- $\lambda$: Decay factor that controls the shape of the curve.

**Steps:**
1. Fit the Nelson-Siegel model to the **spot curve** derived from bond prices.
2. Extract the model parameters $\beta_0$, $\beta_1$, $\beta_2$, and $\lambda$, which summarize the yield curve's shape.

---

### 3. Carry and Roll-Down Returns

#### Carry
The **carry** represents the return earned from holding the bond due to accrued interest, coupon payments, and forward rates. It is calculated as:
$\text{Carry} = \frac{\text{Coupon Payment} + \text{Accrued Interest}}{\text{Initial Price}}$

#### Roll-Down
The **roll-down** is the price appreciation that occurs as the bond "rolls down" the yield curve with the passage of time. It is calculated as:
$\text{Roll-Down} = \frac{\text{Clean Price (Future)} - \text{Clean Price (Today)}}{\text{Clean Price (Today)}}$

By decomposing returns into **carry** and **roll-down**, we isolate the remaining component, often referred to as **excess return**.

---

### 4. Explaining Excess Returns
Once carry and roll-down effects are removed, excess returns can be analyzed using yield curve factors:
$\text{Excess Return} = \text{Observed Return} - (\text{Carry} + \text{Roll-Down})$

**Use of Nelson-Siegel Parameters:**

- **Factors $\beta_0$, $\beta_1$, $\beta_2$:**
  - $\beta_0$: Explains the level of yields (affects long-term bonds).
  - $\beta_1$: Explains the slope of the yield curve (short-term bonds vs. long-term bonds).
  - $\beta_2$: Explains curvature (medium-term fluctuations).

**Regression Analysis:**

Perform cross-sectional regressions to explain excess returns using Nelson-Siegel factors as predictors:
$\text{Excess Return}_i = \alpha + \gamma_1 \beta_0 + \gamma_2 \beta_1 + \gamma_3 \beta_2 + \epsilon_i$
This regression identifies the relationships between yield curve dynamics and bond excess returns.

---

## Workflow

### 1. Spot Curve Fitting
- Use **on-the-run bonds** to construct the spot curve via bootstrapping.
- Fit Nelson-Siegel or Nelson-Siegel-Svensson models to the spot curve.

### 2. Return Decomposition
- Decompose total bond returns into:
  - **Carry**: From accrued interest and forward rates.
  - **Roll-Down**: From price appreciation due to bond maturity shift.

### 3. Excess Return Analysis
- Subtract carry and roll-down from observed returns to isolate excess returns.
- Use regression to explain excess returns with Nelson-Siegel factors.

---

## Usage

### Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
