import collections
import enum
import multiprocessing
import os
import sys
import time
from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing import Queue
from queue import Empty
from typing import NamedTuple
from typing import Optional
from typing import Set

import requests
from bs4 import BeautifulSoup
import validators
from urllib.parse import urlparse

DEFAULT_URL = "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence"
TARGET_DOMAIN = "en.wikipedia.org"
URL_PREFIX = f"https://{TARGET_DOMAIN}"


class Task(NamedTuple):
    url: str


class Result(NamedTuple):
    error_code: int
    source_url: str
    urls: Set[str]


def extract_domain(url: str) -> str:
    domain = urlparse(url).netloc
    return domain[4:] if domain.startswith("www.") else domain


def extract_url(raw_url: Optional[str]) -> Optional[str]:
    if not raw_url:
        return None

    url = f"https://{TARGET_DOMAIN}{raw_url}" if raw_url.startswith("/wiki") else raw_url
    if validators.url(url) and extract_domain(url) == TARGET_DOMAIN:
        return url

    return None


def extract_urls(html_text: str) -> Set[str]:
    extracted_urls = set()

    soup = BeautifulSoup(html_text, 'html.parser')
    for element in soup.find_all("a"):
        extracted_url = extract_url(element.get("href"))
        if extracted_url:
            extracted_urls.add(extracted_url)

    return extracted_urls


def execute(task_queue: Queue, result_queue: Queue, worker_done_queue: Queue, process_id: int) -> None:
    while True:
        terminate = True
        try:
            worker_done_queue.get(block=False)
        except Empty:
            terminate = False

        if terminate:
            break

        task = task_queue.get()
        url = task.url
        try:
            response = requests.get(task.url)
        except Exception as e:
            result_queue.put(Result(error_code=1001, source_url=url, urls=set([])))
            continue

        if response.status_code != 200:
            result_queue.put(Result(error_code=response.status_code, source_url=url, urls=set([])))
            continue

        result_queue.put(Result(error_code=0, source_url=url, urls=extract_urls(response.text)))

    print(f"Terminating process {process_id}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed_url", type=str, default=DEFAULT_URL, help="Initial url to seed crawling")
    parser.add_argument("--logging_frequency", type=int, default=20, help="Logging frequency")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes")
    parser.add_argument("--max_tasks", type=int, default=sys.maxsize, help="Maximum number of urls to crawl")
    args = parser.parse_args()

    task_queue = Queue()
    task_queue.put(Task(url=args.seed_url))
    result_queue = Queue()
    worker_done_queue = Queue()

    processes = []
    for process_id in range(args.num_processes):
        process = Process(target=execute, args=(task_queue, result_queue, worker_done_queue, process_id), daemon=True)
        process.start()
        processes.append(process)
        print(f"Launched process {process_id}")

    urls = {args.seed_url}

    start_time = time.time()
    processed_tasks = 0
    successful_tasks = 0
    recently_crawled_urls = collections.deque(maxlen=5)
    while processed_tasks <= len(urls) and processed_tasks <= args.max_tasks:

        result = result_queue.get()

        for extracted_url in result.urls:
            if extracted_url not in urls:
                urls.add(extracted_url)
                task_queue.put(Task(url=extracted_url))

        processed_tasks += 1
        successful_tasks += result.error_code == 0
        recently_crawled_urls.append(result.source_url)

        if processed_tasks % args.logging_frequency == 0:
            print(
                f"Processed {processed_tasks} tasks\t"
                f"Remaining tasks {len(urls) - processed_tasks}\t"
                f"Throughput {processed_tasks / (time.time() - start_time)} urls/second\t"
                f"Success rate {successful_tasks / processed_tasks}\t"
                f"Recent tasks {recently_crawled_urls}"
            )

    for process_id in range(args.num_processes):
        print(f"Sending {process_id} signal to start")
        worker_done_queue.put(1)

    task_queue.close()
    result_queue.close()


    print("Joining processes")
    for process_id, process in enumerate(processes):
        process.join()
        print(f"Process {process_id} has joined")
    print("Process terminated")


if __name__ == "__main__":
    main()
