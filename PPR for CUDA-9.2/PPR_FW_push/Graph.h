#include <fstream>
#include "Util.cuh"

class Graph {
public:
	// graph topology stored in adjacency lists: source_id->target_id1:target_id2:...
	vector<vector<int>> adj;

	// Construct Graph based on the input file (edgelist or adjacency list)
	Graph(string _data_folder, bool isEdgeList) {
		this->data_folder = _data_folder;
		this->vert_num = 0;
		this->edge_num = 0;
		this->vid_min = INT_MAX;
		this->vid_max = INT_MIN;

		init_graph_attribute();
		this->adj = vector<vector<int>>(this->vert_num, vector<int>());

		if (isEdgeList)	init_graph_topology_edgelist();
		else init_graph_topology_adjacency();

		cout << "INIT--class Graph is constructed" << endl;
		cout << "PREP--graph data in main memory are ready with finally " << this->vert_num 
			 << " vertices [" << this->vid_min << "," << this->vid_max << "] and "
			 << this->edge_num << " edges" << endl;
	}

	~Graph() {
		for (int i = 0; i < this->vert_num; i++) {
			vector<int>().swap(this->adj[i]);
		}
		vector<vector<int>>().swap(this->adj);
		cout << "CLEAR--class Graph is destroyed" << endl;
	}

	// Return the average degree.
	float get_avg_degree() const {
		return (this->edge_num*1.0f) / this->vert_num;
	}

	// Return the minimal vertex id
	int get_vid_min() const {
		return this->vid_min;
	}

	int get_vid_max() const {
		return this->vid_max;
	}

	// Return #vertices
	int get_number_vertices() const {
		return this->vert_num;
	}

	// Return #edges
	int get_number_edges() const {
		return this->edge_num;
	}


/////////////////////////////////////////////////////////////////

private:
	string data_folder;
	int vert_num; //#vertices (id <= 4billion)
	int edge_num; //real #edges  (<= 4billion)
	int vid_min, vid_max;

	// Initialize the number of vertices (v_num) and edges (e_num).
	void init_graph_attribute() {
		string attribute_file = this->data_folder + "/attribute.txt";
		ifstream attr(attribute_file.c_str());
		char c;
		while (true) {
			attr >> c;
			if (c == '=') break;
		}
		attr >> this->vert_num;
		while (true) {
			attr >> c;
			if (c == '=') break;
		}
		attr >> this->edge_num;
		attr.close();

		cout << "INFO--init graph attribute with given " << this->vert_num << " vertices and " 
			<< this->edge_num << " edges" << endl;
	}

	// Initialize the graph topology from input file with edge lists.
	void init_graph_topology_edgelist() {
		// read original topology
		string graph_file = this->data_folder + "/graph.txt";
		fstream f;
		f.open(graph_file);
		
		string line;
		int loc_cur = 0, loc_next = 0;
		int source = 0, target = 0, vert_cnt = 0, edge_cnt = 0;
		vector<bool> vert_exist = vector<bool>(this->vert_num, false);
		while (getline(f, line, '\n')) {
			loc_cur = 0;
			loc_next = 0;
			if (line.empty()) continue;
			
			loc_next = line.find('	', loc_cur);
			if (loc_next == string::npos) continue;
			source = atoi(line.substr(loc_cur, loc_next-loc_cur).c_str());
			this->vid_min = min(this->vid_min, source);
			this->vid_max = max(this->vid_max, source);
			if (!vert_exist[source]) {
				vert_exist[source] = true;
				vert_cnt++;
			}

			loc_cur = loc_next + 1;
			target = atoi(line.substr(loc_cur).c_str());
			if (source != target) {
				this->vid_min = min(this->vid_min, target);
				this->vid_max = max(this->vid_max, target);
				if (!vert_exist[target]) {
					vert_exist[target] = true;
					vert_cnt++;
				}
				this->adj[source].push_back(target);
				edge_cnt++;
			}
		}
		f.close();
		cout << "INFO--init graph topology with really " << vert_cnt << " vertices and " 
			<< edge_cnt << " edges" << endl;
		this->edge_num = edge_cnt;

		//add lost vertices and edges, update # of vertices and edges
		addLostVertsEdges(vert_cnt, edge_cnt, vert_exist);
	}

	// Initialize the graph topology from input file with adjacency lists.
	void init_graph_topology_adjacency() {
		// read original topology
		string graph_file = this->data_folder + "/graph";
		fstream f;
		f.open(graph_file);
		string line;
		int loc_cur = 0, loc_next = 0;
		int source = 0, target = 0, vert_cnt = 0, edge_cnt = 0;
		vector<bool> vert_exist = vector<bool>(this->vert_num, false);
		while (getline(f, line, '\n')) {
			loc_cur = 0;
			loc_next = 0;
			if (line.empty()) continue;
			
			loc_next = line.find('	', loc_cur);
			if (loc_next == string::npos) continue;
			source = atoi(line.substr(loc_cur, loc_next-loc_cur).c_str());
			this->vid_min = min(this->vid_min, source);
			this->vid_max = max(this->vid_max, source);
			if (!vert_exist[source]) {
				vert_exist[source] = true;
				vert_cnt++;
			}

			loc_cur = loc_next + 1;
			loc_next = line.find(':', loc_cur);
			while (loc_next != string::npos) {
				target = atoi(line.substr(loc_cur, loc_next-loc_cur).c_str());
				loc_cur = loc_next + 1;
				loc_next = line.find(':', loc_cur);

				if (source == target) continue;
				this->vid_min = min(this->vid_min, target);
				this->vid_max = max(this->vid_max, target);
				if (!vert_exist[target]) {
					vert_exist[target] = true;
					vert_cnt++;
				}
				this->adj[source].push_back(target);
				edge_cnt++;
			}
			if (loc_cur <= line.size()) {
				target = atoi(line.substr(loc_cur).c_str());
				if (source != target) {
					this->vid_min = min(this->vid_min, target);
					this->vid_max = max(this->vid_max, target);
					if (!vert_exist[target]) {
						vert_exist[target] = true;
						vert_cnt++;
					}
					this->adj[source].push_back(target);
					edge_cnt++;
				}
			}
		}
		f.close();
		cout << "INFO--init graph topology with really " << vert_cnt << " vertices and " 
			<< edge_cnt << " edges" << endl;
		this->edge_num = edge_cnt;

		//add lost vertices and edges, update # of vertices and edges
		addLostVertsEdges(vert_cnt, edge_cnt, vert_exist);
	}

	// Add lost vertices and corresponding edges; for vertices with out-deg 0, add an edge to vid_min
	void addLostVertsEdges(int vert_cnt, int edge_cnt, vector<bool>& vert_exist) {
		if (this->vert_num != vert_cnt) {
			cout << "        ->begin to add lost vertices (vertNum=" << this->vert_num 
				 << ", vertCnt=" << vert_cnt << ")" << endl;
			int vert_add_cnt = 0, edge_add_cnt = 0;
			for (int i = 0; i < this->vert_num; i++) {
				if (!vert_exist[i]) {
					vert_exist[i] = true; //add one vertex with a corresponding edge
					//this->adj[i].push_back(rand()%this->vert_num); //non-determinstic
					this->adj[i].push_back(this->vid_min); //determinstic
					this->vid_min = min(this->vid_min, i); //update vid_min
					this->vid_max = max(this->vid_max, i); //update vid_max
					vert_add_cnt++;
					edge_add_cnt++;
				}
			}
			vert_cnt += vert_add_cnt;
			edge_cnt += edge_add_cnt;
			cout << "        ->add " << vert_add_cnt << " vertices and " << edge_add_cnt << " edges" << endl;
		}

		int edge_add_cnt = 0;
		for (int i = 0; i < this->vert_num; i++) {
			if (this->adj[i].size() == 0) {
				this->adj[i].push_back(this->vid_min);
				edge_add_cnt++;
			}
		}
		edge_cnt += edge_add_cnt;
		if (edge_add_cnt > 0) {
			cout << "        ->add " << edge_add_cnt << " edges to eliminate 0 out-degree" << endl;
		}

		int minMaxCnt = (this->vid_max-this->vid_min) + 1;
		ASSERT(this->vert_num == vert_cnt);
		ASSERT(this->vert_num == minMaxCnt);
		this->vert_num = vert_cnt; //update "vert_num"
		this->edge_num = edge_cnt; //update "edge_num"
	}
};
