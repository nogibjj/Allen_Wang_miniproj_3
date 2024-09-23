from main import (
    read_dataset,
    generate_report,
    generate_summary_statistics,
    create_save_visualization,
)
import os
import pandas as pd


def test_read():
    df = read_dataset(
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )
    # print(df)
    assert type(df) == pd.DataFrame


def test_summary():
    file_path = (
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )
    df = read_dataset(file_path)
    summary_stats, mean_values, median_values, std_dev = generate_summary_statistics(df)
    # those testing mean, median, std value come from excel function

    assert (mean_values["Survived"] - 0.385569335) <= 10 ** (-6)
    assert (mean_values["Pclass"] - 2.305524239) <= 10 ** (-6)
    assert (mean_values["Age"] - 29.47144307) <= 10 ** (-6)
    assert (mean_values["Siblings/Spouses Aboard"] - 0.525366404) <= 10 ** (-6)
    assert (mean_values["Parents/Children Aboard"] - 0.383314543) <= 10 ** (-6)
    assert (mean_values["Fare"] - 32.30542018) <= 10 ** (-6)

    assert median_values["Survived"] == 0
    assert median_values["Pclass"] == 3
    assert median_values["Age"] == 28
    assert median_values["Siblings/Spouses Aboard"] == 0
    assert median_values["Parents/Children Aboard"] == 0
    assert median_values["Fare"] == 14.4542

    assert (std_dev["Survived"] - 0.487004118) <= 10 ** (-6)
    assert (std_dev["Pclass"] - 0.836662004) <= 10 ** (-6)
    assert (std_dev["Age"] - 14.12190841) <= 10 ** (-6)
    assert (std_dev["Siblings/Spouses Aboard"] - 1.104668554) <= 10 ** (-6)
    assert (std_dev["Parents/Children Aboard"] - 0.807465907) <= 10 ** (-6)
    assert (std_dev["Fare"] - 49.7820404) <= 10 ** (-6)


def test_visualization():
    file_path = (
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )
    df = read_dataset(file_path)
    for column in df.columns:
        column_name = column.replace("/", "_")
        create_save_visualization(df, column, column_name + "_distribution.png")
        assert os.path.isfile(column_name + "_distribution.png")


def test_report():
    file_path = (
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )
    df = read_dataset(file_path)
    generate_report(df, "Titanic Profiling Report")
    assert os.path.isfile("Titanic Profiling Report.html")
    assert os.path.isfile("Titanic Profiling Report.md")


if __name__ == "__main__":
    test_read()
    test_summary()
    test_visualization()
    test_report()
