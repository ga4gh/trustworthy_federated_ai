import sys
import time
import tes

TES_CLIENT = tes.HTTPClient("http://localhost:8000")

BUCKET = "fl-models"

def execute_and_wait(task):
    task_id = TES_CLIENT.create_task(task)
    print(f"Task submitted: {task_id}")

    while True:
        status = TES_CLIENT.get_task(task_id, view="FULL")
        print("State:", status.state)

        if status.state == "COMPLETE":
            print("Task completed successfully.")
            return

        if status.state in [
            "EXECUTOR_ERROR",
            "SYSTEM_ERROR",
            "CANCELED",
        ]:
            print(status)
            sys.exit(1)

        time.sleep(2)


def main():

    task = tes.Task(
        name="MinIO Smoke Test",

        inputs=[
            tes.Input(
                url="s3://fl-models/tests/hello.txt",
                path="/inputs/hello.txt",
                type="FILE"
            )
        ],

        executors=[
            tes.Executor(
                image="alpine:latest",
                command=[
                    "sh",
                    "-c",
                    "mkdir -p /outputs && "
                    "cat /inputs/hello.txt > /outputs/result.txt"
                ]
            )
        ],

        outputs=[
            tes.Output(
                path="/outputs/result.txt",
                url="s3://fl-models/tests/result.txt",
                type="FILE"
            )
        ]
    )

    execute_and_wait(task)


if __name__ == "__main__":
    main()