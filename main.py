from tasks.data_init import load_dataset
from tasks.task_1_1 import run_task_1_1
from tasks.task_1_2 import run_task_1_2
from tasks.task_1_3 import run_task_1_3
from tasks.task_1_3_2 import run_task_1_3_2
from tasks.task_1_3_4 import plot_regression
from tasks.task_1_4 import run_task_1_4
from tasks.task_1_5 import run_task_1_5
from tasks.task_1_5_3 import run_task_1_5_3
from tasks.task_1_5_3_plotResiduals import plot_residuals

def main():
    
    df = load_dataset()   # load once

    #run_task_1_1(df)      # Task 1.1
    #run_task_1_2(df)      # Task 1.2
    #run_task_1_3_2(df)      # Task 1.3.2
    #run_task_1_3(df)      # Task 1.3
    #plot_regression(df, "chlorides", -0.100, 5.657, "bilder/chlorides_regression.png")
    #plot_regression(df, "alcohol", 0.390, 5.657, "bilder/alcohol_regression.png")
    #run_task_1_4(df)      # Task 1.4
    #run_task_1_5(df)      # Task 1.5
    run_task_1_5_3(df)    # Task 1.5.3
    plot_residuals(df)  # Task 1.5.3

if __name__ == "__main__": 
    main()
