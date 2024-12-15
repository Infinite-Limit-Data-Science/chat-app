### Runnable

A Runnable represents a unit of work that can transform inputs into outputs. Key Features:

- invoke/ainvoke: supports single input/output processing
- batch/abatch: handles batched inputs for parallel execution
- stream/astream: allows streaming output for single inputs
- astream_log: supports intermediate logging

Note by default, batch runs invoke() in parallel using a thread pool executor. Override to optimize batching.



