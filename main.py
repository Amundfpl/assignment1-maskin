from tasks.data_import import load_dataset
from tasks.task_1_1 import run_task_1_1
from tasks.task_1_2 import run_task_1_2
from tasks.task_1_3 import run_task_1_3

def main():
    df = load_dataset()   # load once

    #run_task_1_1(df)      # Task 1.1
    #run_task_1_2(df)      # Task 1.2
    run_task_1_3(df)      # Task 1.3

if __name__ == "__main__":
    main()
