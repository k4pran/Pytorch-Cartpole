import matplotlib.pyplot as plt


def plot_averages(x, y, title):
    plt.plot(x, y)
    plt.suptitle(title)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
