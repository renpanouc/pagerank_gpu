#include <time.h>
#include "DeviceMemory.cuh"

// Initialize entry values in device memory
template<typename ValueType>
__global__ void InitPPR(ValueType *pagerank, ValueType *residual, ValueType *messages, 
	bool *isactive, int *active_verts, int *active_verts_num, 
	const int source, const int vert_num);

// Update pagerank
template<typename ValueType>
__global__ void UpdateVertex(const int *active_verts, const int *active_verts_num, 
	const ValueType *residual, ValueType *pagerank, const ValueType alpha);

// Active vertices generate new messages and then push messages to target vertices
template<typename ValueType>
__global__ void PushMessage(const int *active_verts, const int *active_verts_num, 
	const int *row_ptr, const int *col_ind, const ValueType alpha, 
	ValueType *residual, ValueType *messages, bool *isactive);

// Prepare for the next iteration
template<typename ValueType>
__global__ void Barrier(const int *row_ptr, 
	const int vert_num, const ValueType alpha, const ValueType rmax, 
	int *active_verts, int *active_verts_num, 
	ValueType *residual, ValueType *messages, bool *isactive);

// Dump results
void DumpResults(const int verts_num, ValueType *d_pagerank);

int main() {
	// Initialize graph data in host & device memory
	string dir = "/home/w/wangning/data/livej";
	Graph graph(dir, false);
	//Graph graph(dir, true);
	
	DeviceMemory device_memory(graph.get_number_vertices(), graph.get_number_edges());
	device_memory.CudaMemcpyGraph(graph);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cout << "\n==================== PPR with FORWARD PUSH starts ====================" << endl;

	// Initialize parameters for PPR
	int source = 1, vert_num = graph.get_number_vertices();
	ValueType alpha = 0.2f;
	ValueType rmax = 0.1f*(1.0f/graph.get_number_edges());
	CUDA_ERROR(cudaMemset(device_memory.active_verts_num, 0, sizeof(int)));
	InitPPR<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(
		device_memory.pagerank, device_memory.residual, device_memory.messages, 
		device_memory.isactive, device_memory.active_verts, device_memory.active_verts_num, 
		source, vert_num);
	
	unsigned long total_updates = 0;
	int blocks_num = 0;
	int active_verts_num = 0;
	int iteration_cnt = 1;
	
	// Iterative computations start.
	while (1) {
		CUDA_ERROR(cudaMemcpy(&active_verts_num, device_memory.active_verts_num, 
					sizeof(int), cudaMemcpyDeviceToHost));
		std::cout << "PPR--ite=" << iteration_cnt << "\tactive=" << active_verts_num << endl;
		if (active_verts_num == 0) break;
		total_updates += active_verts_num;
		blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, active_verts_num);
		
		//re-calculate #blocks based on "active_verts_num" so that threads can correctly find 
		//the corresponding active vertices in "active_verts" constructed in InitPPR() or Barrier() 
		UpdateVertex<ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(
			device_memory.active_verts, device_memory.active_verts_num, 
			device_memory.residual, device_memory.pagerank, alpha);

		PushMessage<ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(
			device_memory.active_verts, device_memory.active_verts_num, 
			device_memory.row_ptr, device_memory.col_ind, alpha, 
			device_memory.residual, device_memory.messages, device_memory.isactive);

		CUDA_ERROR(cudaMemset(device_memory.active_verts_num, 0, sizeof(int)));
		Barrier<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(
			device_memory.row_ptr, vert_num, alpha, rmax, 
			device_memory.active_verts, device_memory.active_verts_num,
			device_memory.residual, device_memory.messages, device_memory.isactive);

		iteration_cnt++;
	}
	cout << "total updates: " << total_updates << endl;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float runtime = 0; //milliseconds
	cudaEventElapsedTime(&runtime, start, stop);
	cout << "gpu runtime: " << runtime/1000.0 << " seconds" << endl;
	cout << "==================== PPR with FORWARD PUSH ends ====================\n" << endl;

	DumpResults(graph.get_number_vertices(), device_memory.pagerank);
	return 0;
}

// Initialize entry values
template<typename ValueType>
__global__ void InitPPR(ValueType *pagerank, ValueType *residual, ValueType *messages, 
		bool *isactive, int *active_verts, int *active_verts_num, 
		const int source, const int vert_num) {
	size_t thread_id = threadIdx.x;
	size_t schedule_offset = blockDim.x * blockIdx.x;
	size_t vid = 0;
	while (schedule_offset < vert_num) {
		vid = schedule_offset + thread_id;
		//in the last batch, some threads are idle
		if (vid < vert_num) {
			pagerank[vid] = 0;
			residual[vid] = 0;
			isactive[vid] = false;
			messages[vid] = 0;
		}
		schedule_offset += blockDim.x * gridDim.x;
	}

	//prepare for the 1st iteration
	size_t global_id = thread_id + blockDim.x*blockIdx.x;
	if (global_id == 0) {
		residual[source] = 1;
		*active_verts_num = 1;
		active_verts[0] = source;
	}
}

// Update pagerank
template<typename ValueType>
__global__ void UpdateVertex(const int *active_verts, const int *active_verts_num, 
		const ValueType *residual, ValueType *pagerank, const ValueType alpha) {
	size_t thread_id = threadIdx.x;
	size_t schedule_offset = blockDim.x * blockIdx.x;
	size_t idx = 0;
	int u = 0;
	int total_active_verts_num = *active_verts_num;
	while (schedule_offset < total_active_verts_num) {
		idx = schedule_offset + thread_id;
		//in the last batch, some threads are idle
		if (idx < total_active_verts_num) {
			u = active_verts[idx];
			pagerank[u] += alpha * residual[u];
		}
		schedule_offset += blockDim.x * gridDim.x;
	}
}

// Active vertices generate new messages and then push messages to target vertices
template<typename ValueType>
__global__ void PushMessage(const int *active_verts, const int *active_verts_num, 
		const int *row_ptr, const int *col_ind, const ValueType alpha, 
		ValueType *residual, ValueType *messages, bool *isactive) {
	size_t thread_id = threadIdx.x;
	size_t lane_id = thread_id % THREADS_PER_WARP; //the local id in a warp (from 0)
	size_t warp_id = thread_id / THREADS_PER_WARP; //the i-th warp (from 0)

	typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
	__shared__ typename BlockScan::TempStorage block_temp_storage; //shared memory by threads per block

	//[0]->thread/lane_id; [1/2]->start/end offset
	volatile __shared__ int comm[THREADS_PER_BLOCK/THREADS_PER_WARP][3];
	volatile __shared__ ValueType commr[THREADS_PER_BLOCK/THREADS_PER_WARP];
	volatile __shared__ int comm2[THREADS_PER_BLOCK]; 
	volatile __shared__ ValueType commd2[THREADS_PER_BLOCK]; //out-degree
	volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];

	int total_active_verts_num = *active_verts_num;
	size_t schedule_offset = blockDim.x * blockIdx.x;
	size_t idx = 0;
	int row_start, row_end;
	int u, v; //id of a source vertex that will push messages to targets; id of a target vertex
	ValueType ru, msg; //value of "u"
	
	// <0>-while: there are (blockDim.x * gridDim.x) vertices scheduled in one loop
	//            since we have totally blockDim.x * gridDim.x threads
	while (schedule_offset < total_active_verts_num) {
		idx = schedule_offset + thread_id;
		if (idx < total_active_verts_num) {
			u = active_verts[idx]; //an active vertex owned by this thread
			ru = residual[u]; //value of "u"
			residual[u] = 0;
			row_start = row_ptr[u]; //start offset of outgoing edges of "u" in "col_ind"
			row_end = row_ptr[u+1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
		} else {
			row_start = 0;
			row_end = 0;
		}

		// <1>-while: block-granularity coarse grained pushing
		//            Among (blockDim.x*gridDim.x) active vertices scheduled in this <0>-while loop, all 
		//            vertices with out-degree greater than or at least equal to THREADS_PER_BLOCK will 
		//            be processed in <1>-while. 
		//            todo_edges_num = row_end - row_start;
		//            _syncthreads_or(x): synchronize threads per block and return the result of performing 
		//                                "or" over "x" owned by this thread in the same block
		while (__syncthreads_or((row_end-row_start)>=THREADS_PER_BLOCK)) {
			if ((row_end-row_start) >= THREADS_PER_BLOCK) {
				comm[0][0] = thread_id; //I (thread_id) want to process the active vertex assigned to me.
			}
			__syncthreads(); //all threads in one block vote to processing their own vertices
			
			if (comm[0][0] == thread_id) {
				comm[0][1] = row_start; //the vertx owned by me will be processed in this <1>-while loop.
				comm[0][2] = row_end;
				commr[0] = ru;
				row_start = row_end; //avoid processing this vertex repeatedly in <2>&<3>-while
			}
			__syncthreads(); //all threads are ready to process the selected vertex

			size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
			size_t push_ed = comm[0][2];
			// <1.1>-while: block-granularity-outgoing edges
			//              In each <1.1>-while loop, select THREADS_PER_BLOCK outgoing edges for the vertex 
			//              scheduled in this <1>-while loop to be processed by THREADS_PER_BLOCK threads in 
			//              this block. One thread corresponds to one edge. 
			//              Some threads might be idle in the last <1.1>-while loop.
			while (__syncthreads_or(push_st<push_ed)) {
				if (push_st < push_ed) {
					v = col_ind[push_st]; //target vertex id
					msg = ((1-alpha)*commr[0]) / (comm[0][2]-comm[0][1]); //outdeg of the selected s, not "u"
					atomicAdd(messages+v, msg); //residual[v]+=msg; rv=the old residual[v]+msg
					isactive[v] = true;
				}
				push_st += THREADS_PER_BLOCK; //until all outgoing edges of "u" have been processed
			}
		} //until all source vertices with "todo_edges_num>=THREADS_PER_BLOCK" have been processed
		

		// <2>-while: warp(32)-medium granularity pushing
		//            Among (blockDim.x*gridDim.x) active vertices scheduled in this <0>-while loop, all 
		//            vertices with out-degree greater than or at least equal to THREADS_PER_WARP(32) but 
		//            less than THREADS_PER_BLOCK will be processed in <2>-while. 
		//            todo_edges_num = row_end - row_start;
		//            WarpAny(x): synchronize threads per warp and return the result of performing 
		//                        "or" over "x" owned by this thread in the same warp
		//            Note: the multiple warps in the same block run at the same time
		while (__any_sync(FULL_MASK, (row_end-row_start)>=THREADS_PER_WARP)) {
			if ((row_end-row_start) >= THREADS_PER_WARP) {
				comm[warp_id][0] = lane_id; //threads in the "warp_id"-th warp try to vote
			}
			//threads in the same warp can naturally synchronize with each other
			//_synchronize() is only used to synchronize threads in the whole block
			if (comm[warp_id][0] == lane_id) {
				comm[warp_id][1] = row_start; //vertex owned by the "lane_id"-th thread in a warp is scheduled
				comm[warp_id][2] = row_end;
				commr[warp_id] = ru;
				row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
			}
			size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
			size_t push_ed = comm[warp_id][2];
			// <2.1>-while: warp-granularity-outgoing edges
			//        		In each <2.1>-while loop, select THREADS_PER_WARP outgoing edges for the vertex 
			//              scheduled in this <2>-while loop to be processed by THREADS_PER_WARP threads in 
			//              this block. One thread corresponds to one edge. 
			//              Some threads might be idle in the last <1.1>-while loop.      
			while (__any_sync(FULL_MASK, push_st<push_ed)) {
				if (push_st < push_ed) {
					v = col_ind[push_st];
					//use the out-degree of the selected source, not "u"
					msg = ((1-alpha)*commr[warp_id]) / (comm[warp_id][2]-comm[warp_id][1]); 
					atomicAdd(messages+v, msg);
					isactive[v] = true;
				}
				push_st += THREADS_PER_WARP; //until all outgoing edges of "u" have been processed
			}
		} //until all source vertices with "todo_edges_num>=THREADS_PER_WARP" have been processed
		

		//then, the out-degree of "u" is less than THREADS_PER_WARP(32)
		int thread_count = row_end - row_start;
		int deg = thread_count;
		int scatter = 0, total = 0;
		__syncthreads();
		BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total); 
		__syncthreads(); //there are "total" edges left in every block

		int progress = 0;
		// <3>-while: fine-grained pushing
		//            Among (blockDim.x*gridDim.x) active vertices scheduled in this <0>-while loop, all 
		//            vertices with out-degree less than THREADS_PER_WARP(32) will be processed in <3>-while. 
		//            todo_edges_num/thread_count = row_end - row_start;
		while (progress < total) {
			//threads in one block have the same "total", "progress", "remain" and "cur_batch_count" values
			//threads per block fill up "comm2" & "commr2" with THREADS_PER_BLOCK elements
			//because deg=row_end-row_start, we have "deg" replicas of a vertex value in "commr2"
			//only the thread with highest thread_id in one block may have 
			//"scatter>=(progress+THREADS_PER_BLOCK)"
			int remain = total - progress;
			while (scatter<(progress+THREADS_PER_BLOCK) && (row_start<row_end)) {
				comm2[scatter-progress] = row_start;
				commd2[scatter-progress] = deg; //record the out-degree of the selected source vertex
				commr2[scatter-progress] = ru;
				scatter++;
				row_start++;
			}
			__syncthreads();
			int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
			if (thread_id < cur_batch_count) {
				v = col_ind[comm2[thread_id]];
				msg = ((1-alpha)*commr2[thread_id]) / commd2[thread_id]; //use the correct out-degree
				atomicAdd(messages+v, msg);
				isactive[v] = true;
			}
			progress += THREADS_PER_BLOCK;
		}

		//schedule (blockDim.x * gridDim.x) active vertices per <0>-while loop
		schedule_offset += blockDim.x * gridDim.x;
	}
}

// Prepare for the next iteration.
template<typename ValueType>
__global__ void Barrier(const int *row_ptr, 
		const int vert_num, const ValueType alpha, const ValueType rmax, 
		int *active_verts, int *active_verts_num, 
		ValueType *residual, ValueType *messages, bool *isactive) {
	typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
	__shared__ typename BlockScan::TempStorage block_temp_storage;
	volatile __shared__ size_t output_cta_offset;

	size_t thread_id = threadIdx.x;
	size_t schedule_offset = blockDim.x * blockIdx.x;
	size_t vid = 0;
	while (schedule_offset < vert_num) {
		vid = schedule_offset + thread_id;
		int thread_cnt = 0;
		//in the last batch, some threads are idle
		if (vid < vert_num) {
			if (isactive[vid]) {
				residual[vid] += messages[vid]; //not "=", to accumulate received messages
				messages[vid] = 0;
				isactive[vid] = false;
				
				if ((residual[vid]/(row_ptr[vid+1]-row_ptr[vid])) >= rmax) {
					thread_cnt = 1;
				}
			}
		}

		int scatter = 0, total = 0;
		__syncthreads();
		BlockScan(block_temp_storage).ExclusiveSum(thread_cnt, scatter, total);
		__syncthreads();
		if (thread_id == 0) {
			output_cta_offset = atomicAdd(active_verts_num, total); //run per block
		}
		__syncthreads();
		if (thread_cnt > 0) {
			active_verts[output_cta_offset+scatter] = vid;
		}

		schedule_offset += blockDim.x * gridDim.x;
	}
}

// Dump results
void DumpResults(const int verts_num, ValueType *d_pagerank) {
	ValueType *h_pagerank = new ValueType[verts_num];
	CUDA_ERROR(cudaMemcpy(h_pagerank, d_pagerank, 
				verts_num*sizeof(ValueType), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 20; i++) {
		if (h_pagerank[i] > 0) {
			cout << i << ", " << h_pagerank[i] << endl;
		}
	}
	delete[] h_pagerank;
	h_pagerank = NULL;
}
