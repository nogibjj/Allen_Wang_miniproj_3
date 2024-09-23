import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def read_dataset(file_path):
    df = None
    if file_path.endswith(".csv"):
        df = pl.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pl.read_excel(file_path)  # Polars supports Excel reading
    return df


def generate_summary_statistics(df):
    # Using polars to generate summary statistics
    numeric_df = df.select(pl.col(pl.Float64) | pl.col(pl.Int64))
    
    summary_stats = numeric_df.describe()
    mean_values = numeric_df.mean()
    median_values = numeric_df.median()
    std_dev = numeric_df.std()
    return summary_stats, mean_values, median_values, std_dev


def create_save_visualization(df, column_name, save_filename=None, show=False):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    # Convert to numpy for visualization in seaborn
    sns.histplot(df[column_name].to_numpy(), kde=True, color="skyblue", bins=30)
    plt.title(f"{column_name} Distribution", fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    if show:
        plt.show()


def generate_report(df, title):
    # Generate summary statistics using polars
    summary_stats, mean_values, median_values, std_dev = generate_summary_statistics(df)
    
    with open(title + ".md", "w", encoding="utf-8") as file:
        file.write("Summary:\n")
        file.write(str(summary_stats) + "\n\n")
        file.write("Mean:\n")
        file.write(str(mean_values) + "\n\n")
        file.write("Median:\n")
        file.write(str(median_values) + "\n\n")
        file.write("Standard Deviation:\n")
        file.write(str(std_dev) + "\n\n")
        file.write("![image1](Age_distribution.png)\n")
        file.write("\n\n")
        file.write("![image2](Fare_distribution.png)\n")
        file.write("\n\n")
        file.write("![image3](Pclass_distribution.png)\n")

