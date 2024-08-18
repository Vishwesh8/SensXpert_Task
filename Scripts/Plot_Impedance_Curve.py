from utilities import *

# Add path to the folder that contains data here
path = r"E:\MS in Robosys\Job applications\New Job Applications\sensXPERT\Task\data"
files = glob.glob(path + "/*.csv")
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
