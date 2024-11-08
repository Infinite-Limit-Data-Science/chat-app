TGI server deployment can easily be monitored through a Grafana dashboard, consuming a Prometheus data collection. Example of inspectable metrics are statistics on the effective batch sizes used by TGI, prefill/decode latencies, number of generated tokens, etc.

First, on your server machine, TGI needs to be launched as usual. TGI exposes multiple metrics that can be collected by Prometheus monitoring server.

Here is a list of all exported metrics on the Prometheus scrapable/metrics route:

1) Input Summaries
- tgi_request_input_length_bucket: prom summary of the number of tokens inside each request prompt
- tgi_request_max_new_tokens_bucket: prom summary of the total token size of each request
- tgi_request_generated_tokens_bucket: prom summary of the number of generated tokens for each request
- tgi_batch_next_size_bucket: prom summary of the size of each new batch from the queue

2) Latency Summaries
- tgi_request_duration_bucket: prom summary of the total end to end duration for each request
- tgi_request_validation_duration_bucket: prom summary of the duration of the validation step per request
- tgi_request_queue_duration_bucket: prom summary of the time spent waiting in the internal queue before a request is added to a batch
- tgi_request_inference_duration_bucket: prom summary of inference duration for each request
- tgi_request_mean_time_per_token_duration_bucket: prom summary of the inference duration for each token of a request
- tgi_batch_concat_duration_bucket: prom summary of the batch concatenation duration.
- tgi_batch_forward_duration_bucket: prom summary of the model forward duration for each batch. You can filter by method = [“prefill”, “decode”]
- tgi_batch_decode_duration_bucket: prom summary of the model decode token step duration for each batch. You can filter by method = [“prefill”, “decode”]
- tgi_batch_filter_duration_bucket: prom summary of the batch filtering duration for each batch. You can filter by method = [“prefill”, “decode”]
- tgi_batch_inference_duration_bucket: prom summary of the total duration of inference (optional concat, forward, decode token step and filter) for each batch. You can filter by method = [“prefill”, “decode”]

3) Counters
- tgi_request_count: prom counter of the total number of request received.
- tgi_request_success: prom counter of the number of request that suceeded.
- tgi_request_failure: prom counter of the number of failed requests. You can filter by err.
- tgi_batch_concat: prom counter of the number of batch concatenates. You can filter by reason = [“backpressure”, “wait_exceeded”]
- tgi_batch_inference_success: prom counter of the number of inference that suceeded.
- tgi_batch_inference_failure: prom counter of the number of inference that failed. You can filter by method = [“prefill”, “decode”]
- tgi_batch_inference_count: prom counter of the total number of inference.

4) Gauges
- tgi_queue_size: prom gauge of the number of requests waiting in the internal queue
- tgi_batch_current_max_tokens: prom gauge of the number of maximum tokens the current batch will grow to
- tgi_batch_current_size: prom gauge of the current batch size

In the rest of this tutorial, we assume that TGI was launched through Docker with --network host.




https://huggingface.co/docs/text-generation-inference/main/en/reference/metrics

https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/monitoring