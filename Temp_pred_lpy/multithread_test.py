"""
Created on 8/6/2019
@author: no281
"""
import threading
import time


def job():
    # print(1000**1000**2)
    time.sleep(1)
    print("当前线程的个数:", threading.active_count())
    print("当前线程的信息:", threading.current_thread())

if __name__ == '__main__':
    thread_nums = 8
    thread_list=[]
    for i in range(thread_nums):
        t = threading.Thread(target=job,name='job'+str(i),args=())
        thread_list.append(t)

    for t in thread_list:
        t.start()
