import numpy as np
import matplotlib.pyplot as plt


def my_linfit(x, y):
    # Initialize slope and intercept.
    a = 0
    b = 0

    # There is one point or many duplicate points.
    if len(x) == 1 or (len(set(x)) == 1 and len(set(y)) == 1):
        a = 0
        b = y[0]
    else:
        # Calculate the slope.
        a = (np.mean(x * y) - np.mean(x) * np.mean(y)) / \
            (np.mean(x ** 2) - np.mean(x) ** 2)

        # Calculate intercept.
        b = (np.mean(x ** 2) * np.mean(y) - np.mean(x) * np.mean(x * y)) / \
            (np.mean(x ** 2) - np.mean(x) ** 2)

    return a, b


def calculate_error(y_d, y_p):
    # y_d are collected data points.
    # y_p are predicted y values from x values of
    # collected data points.

    res = y_d - y_p
    mae = np.mean(abs(res))
    print(f"Mean Absolute Error (MAE): {mae}")
    mse = np.mean(res ** 2)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Square root of mean square error: {np.sqrt(mse)}\n")


# Get clicked point then draw the line and print error information.
def onclick(event):
    # Left click for collecting points.
    if event.button == 1:
        global ix, iy
        ix, iy = event.xdata, event.ydata

        global coords
        coords.append((ix, iy))
        plt.plot(ix, iy, 'o')
        fig.canvas.draw()
    # Right click for stop collecting points.
    elif event.button == 3:
        if len(coords) == 0:
            print("Please press more times.")
        else:
            # Extract x and y coordinates.
            x = np.array([c[0] for c in coords])
            y = np.array([c[1] for c in coords])

            # Calculate the slope and intercept that minimize mean square error.
            a, b = my_linfit(x, y)

            # Predicted y values.
            y_p = a * x + b

            # Evaluate model
            calculate_error(y, y_p)

            # Draw collected points.
            plt.plot(x, y, 'o')

            # Draw predicted line.
            t = np.linspace(-10, 100, 100)
            y_draw = a * t + b
            plt.grid(True)
            plt.plot(t, y_draw)
            plt.title(f"Fitted line and {len(coords)} data points")
            plt.show()
    else:
        return


fig = plt.figure()
plt.title('Press mouse many times and then stop the by right click')
plt.axis([0, 100, 0, 1000])
plt.grid(True)
coords = []

# Get points and draw.
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


