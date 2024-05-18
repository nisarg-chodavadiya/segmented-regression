import numpy as np
import matplotlib.pyplot as plt
from segmented_regressor import SegmentedRegressor

# Example usage
if __name__ == "__main__":
    # Generate some synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = np.piecewise(X, [X < 4, (X >= 4) & (X < 7), X >= 7], [lambda x: 2 + 0.5*x, lambda x: 4 + 1.5*x, lambda x: 7 - 0.5*x])
    y += np.random.normal(scale=0.5, size=y.shape)

    # Define breakpoints
    breakpoints = [4, 7]

    # Fit segmented regression model
    model = SegmentedRegressor(breakpoints)
    model.fit(X, y)

    # Predict and print summary
    y_pred = model.predict(X)
    model.summary()

    # Plotting
    plt.scatter(X, y, label='Data')
    plt.plot(X, y_pred, color='red', label='Segmented Regression')
    for bp in breakpoints:
        plt.axvline(bp, color='green', linestyle='--', label=f'Breakpoint at {bp}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()