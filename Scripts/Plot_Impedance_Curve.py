from utilities import *

# Get current directory
cwd = os.getcwd()
# Get parent directory
task_dir = os.path.abspath(os.path.join(cwd, os.pardir))
# Get data directory
data_dir = os.path.join(task_dir, "data")

files = glob.glob(data_dir + "/*.csv")
files = random.sample(files, 10)

# loop over the list of csv files
for f in files:
    # reading the csv file
    df = pd.read_csv(f, encoding='unicode_escape', sep="\s|:", engine='python')

    # Keeping only required columns and converting data to required format
    df = df.iloc[:, :4]
    df = df.applymap(lambda x: float(x.replace(',', '.').replace('E', 'e')))

    # Finding critical points and creating dataset
    cp2, cp3, cp4 = critical_points(df)

    # Plotting impedance curve for this cycle
    cycle = f.split('\\')[-1].split('.')[0]
    impedance_curve(df, cycle, cp2, cp3, cp4)
