import mteb
from mteb.task_selection import results_to_dataframe

import json
import os
import sys
import csv


for root in ["your_model_name"]: 
    path = 'results/local/'+root+'/local__'+root+"/no_revision_available"
    results_list = os.listdir(path)
    print(results_list)
    benchmark = "MTEB(Multilingual, v2)"
    if len(sys.argv) > 1:
        benchmark = sys.argv[1]
    results = {}
    split_tasks = {}
    task_types_map = {}  # 用于记录每个任务的类型

    def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
        if benchmark:
            tasks = mteb.get_benchmark(benchmark).tasks
        else:
            tasks = mteb.get_tasks(languages=languages, tasks=names)

        return tasks

    tasks = get_tasks(names=None, languages=None, benchmark=benchmark)
    names = [t.metadata.name for t in tasks]
    tasks = {name: task for name, task in zip(names, tasks)}

    # print('names', names)
    for task in results_list:
        if task.split(".json")[0] not in names:
            continue
        name = task.split(".json")[0]
        meta = tasks[name].metadata 
        with open(os.path.join(path, task)) as f:
            result = json.load(f)
        # print('result', result)
        task_type = meta.type
        eval_split = list(result['scores'].keys())[0]
        
        score = sum([ele['main_score'] for ele in result['scores'][eval_split]]) / len(result['scores'][eval_split])
        results[name] = round(score * 100, 2)
        task_types_map[name] = task_type
        if task_type not in split_tasks:
            split_tasks[task_type] = []
        split_tasks[task_type].append(score)

    final_scores = sum(results.values()) / len(results)
    missed_tasks = [name for name in names if name not in results]
    print('missed tasks', missed_tasks)
    print('final score', len(results), final_scores)
    scores = []
    for task_type in split_tasks:
        print(task_type, len(split_tasks[task_type]), sum(split_tasks[task_type]) / len(split_tasks[task_type]))
        score = sum(split_tasks[task_type]) / len(split_tasks[task_type])
        scores.append(score)
    print('Mean(Type)', sum(scores) / len(scores))

    for name in results:
        print(name, results[name])

    # 将结果写入CSV文件

    with open('results/local/'+root+'/local__'+root+ "/" + benchmark +'_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Task Name', 'Task Type', 'Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for name in sorted(results.keys()):
            writer.writerow({
                'Task Name': name,
                'Task Type': task_types_map[name],
                'Score': results[name]
            })
        
        # 添加汇总行
        writer.writerow({})  # 空行
        writer.writerow({
            'Task Name': 'Final Score (Overall)',
            'Task Type': '-',
            'Score': round(final_scores, 2)
        })
        
        # 添加各任务类型的平均分
        for task_type in sorted(split_tasks.keys()):
            type_score = sum(split_tasks[task_type]) / len(split_tasks[task_type])
            writer.writerow({
                'Task Name': f'Mean({task_type})',
                'Task Type': task_type,
                'Score': round(type_score * 100, 2)
            })

    print(f'\n结果已保存到 {csvfile}')