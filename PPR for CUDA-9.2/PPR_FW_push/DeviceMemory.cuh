#include "Graph.h"

class DeviceMemory {
public:
	/**
	* graph variables
	*   input graph in CPU "source->(target1,target2,...)": 0->(2,3); 1->(0,3,4,5); ...
	*   input graph in GPU:
	*     outgoing edge array (targets)->col_ind: 2 3  0 3 4 5 ...
	*                                             |   /
	*                                             |  /
	*                     pointer array->row_prt: 0 2 ...
	*                      array index (sources): 0 1 ...
	*/
	int *row_ptr;
	int *col_ind;

	/**
	 * application variables for personalized pagerank (forward push)
	 */
	ValueType *pagerank; //i.e., reserve
	ValueType *residual;
	ValueType *messages;
	bool      *isactive;
	int *active_verts; //active vertices in the current iteration
	int *active_verts_num; //number of act verts in the current iteration


	DeviceMemory(int _v_num, int _e_num):vert_num(_v_num), edge_num(_e_num) {
		row_ptr = NULL;
		col_ind = NULL;

		pagerank = NULL;
		residual = NULL;
		messages = NULL;
		isactive = NULL;
		active_verts = NULL;
		active_verts_num = NULL;

		CudaMallocData();
		cout << "INIT--class DeviceMemory is constructed" << endl;
	}

	~DeviceMemory() {
		CudaFreeData();
		cout << "CLEAR--class DeviceMemory is destroyed" << endl;
	}

	// Copy graph data from host to device.
	void CudaMemcpyGraph(Graph& graph){
		CudaMemcpyRowPtr(graph);
		CudaMemcpyColInd(graph);
		cout << "PREP--graph data in device memory are now ready" << endl;
	}


private:
	int vert_num;
	int edge_num;

	// Allocate memory for data in GPU.
	void CudaMallocData() {
		// graph
		CUDA_ERROR(cudaMalloc(&row_ptr, sizeof(int)*(vert_num+1)));
		CUDA_ERROR(cudaMalloc(&col_ind, sizeof(int)*edge_num));
		
		// application
		CUDA_ERROR(cudaMalloc(&pagerank, sizeof(ValueType)*vert_num));
		CUDA_ERROR(cudaMalloc(&residual, sizeof(ValueType)*vert_num));
		CUDA_ERROR(cudaMalloc(&messages, sizeof(ValueType)*vert_num));
		CUDA_ERROR(cudaMalloc(&isactive, sizeof(bool)*vert_num));
		CUDA_ERROR(cudaMalloc(&active_verts, sizeof(int)*vert_num));
		CUDA_ERROR(cudaMalloc(&active_verts_num, sizeof(int)));
	}

	// Release memory for data in GPU.
	void CudaFreeData() {
		// graph
		if (row_ptr) CUDA_ERROR(cudaFree(row_ptr));
		if (col_ind) CUDA_ERROR(cudaFree(col_ind));

		// application
		if (pagerank) CUDA_ERROR(cudaFree(pagerank));
		if (residual) CUDA_ERROR(cudaFree(residual));
		if (messages) CUDA_ERROR(cudaFree(messages));
		if (isactive) CUDA_ERROR(cudaFree(isactive));
		if (active_verts) CUDA_ERROR(cudaFree(active_verts));
		if (active_verts_num) CUDA_ERROR(cudaFree(active_verts_num));
	}

	// Create indices of edges and then copy them from host to device.
	void CudaMemcpyRowPtr(Graph& graph){
		int *rptr = new int[vert_num+1];
		for (int i = 0; i < vert_num; i++) rptr[i] = graph.adj[i].size();
		for (int i = 0, prefix = 0; i < (vert_num+1); i++) {
			int tmp = rptr[i];
			rptr[i] = prefix;
			prefix += tmp;
		}
		CUDA_ERROR(cudaMemcpy(row_ptr, rptr, sizeof(int)*(vert_num+1), cudaMemcpyHostToDevice));
		delete[] rptr;
		rptr = NULL;
	}

	// Copy edges from host to device.
	void CudaMemcpyColInd(Graph& graph){
		int *cind = new int[edge_num];
		int prev_edge_num = 0;
		int neighbor_num = 0;
		for (int i = 0; i < vert_num; i++){
			neighbor_num = graph.adj[i].size();
			for (int j = 0; j < neighbor_num; j++){
				int off = prev_edge_num + j;
				cind[off] = graph.adj[i][j];
			}
			sort(cind+prev_edge_num, cind+prev_edge_num+neighbor_num);
			prev_edge_num += neighbor_num;
		}
		CUDA_ERROR(cudaMemcpy(col_ind, cind, sizeof(int)*edge_num, cudaMemcpyHostToDevice));
		delete[] cind;
		cind = NULL;
	}
};