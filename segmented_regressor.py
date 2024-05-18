import numpy as np
from scipy.optimize import minimize

class SegmentedRegressor:
    def __init__(self, breakpoints):
        self.breakpoints = sorted(breakpoints)
        self.coefficients = []
        self.fitted = False

    def _linear_model(self, X, params):
        """Compute the piecewise linear regression model."""
        segments = np.split(X, np.searchsorted(X, self.breakpoints))
        y_pred = []
        param_idx = 0
        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue
            intercept, slope = params[param_idx:param_idx+2]
            y_pred.extend(intercept + slope * segment)
            param_idx += 2
        return np.array(y_pred)

    def _objective_function(self, params, X, y):
        """Objective function to minimize: Sum of squared residuals."""
        y_pred = self._linear_model(X, params)
        residuals = y - y_pred
        return np.sum(residuals**2)

    def fit(self, X, y):
        """Fit the segmented regression model to the data."""
        X = np.asarray(X)
        y = np.asarray(y)

        # Initial parameters: zeros for intercepts and slopes
        initial_params = np.zeros(2 * (len(self.breakpoints) + 1))

        # Optimize the parameters
        result = minimize(self._objective_function, initial_params, args=(X, y))
        self.coefficients = result.x
        self.fitted = True

    def predict(self, X):
        """Predict using the fitted segmented regression model."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call the 'fit' method first.")
        X = np.asarray(X)
        return self._linear_model(X, self.coefficients)

    def summary(self):
        """Print the summary of the fitted model."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call the 'fit' method first.")

        print("Segmented Regression Model Summary")
        print("=================================")
        segments = len(self.breakpoints) + 1
        for i in range(segments):
            intercept, slope = self.coefficients[2*i:2*i+2]
            if i == 0:
                print(f"Segment 1 (X <= {self.breakpoints[i]}): Intercept = {intercept:.4f}, Slope = {slope:.4f}")
            elif i == segments - 1:
                print(f"Segment {segments} (X > {self.breakpoints[i-1]}): Intercept = {intercept:.4f}, Slope = {slope:.4f}")
            else:
                print(f"Segment {i+1} ({self.breakpoints[i-1]} < X <= {self.breakpoints[i]}): Intercept = {intercept:.4f}, Slope = {slope:.4f}")

