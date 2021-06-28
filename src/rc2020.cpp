#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <math.h>
#include <pthread.h>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "gurobi_c++.h"

using namespace rapidjson;
using namespace std;

// macros for seasons
#define FULL_SEASON 0
#define WINTER_SEASON 1
#define SUMMER_SEASON 2
#define INTER__SEASON 3

// gurobi environment object, we make it global because we only need one
GRBEnv env = GRBEnv();

// number of threads used for local search
#define NUM_THREADS 4

// path to solution file, program starting time and execution time limit
// global since they will be used in several procedures
string solutionfilename = "sol.txt";
long int programStartTime;
int timeLimitInSeconds = 15 * 60;

// random number generator - different for each thread
void srand_multithread(int g, int threadID);
long rand_multithread(int threadid = 0);

// sometimes we ignore some scenarios in excess calculation
// in order to be able to solve the remaining problem with quantile 1 using MIP
vector<vector<bool> > isScenarioRemovedAtTS(365, vector<bool>(700, false));

// returns an element at position 'index' when array 'risks' is sorted
// used for calculating quantile, faster than sorting
double select(vector<double>& risks, const int index)
{
	std::nth_element(risks.begin(), risks.begin() + index, risks.end());
	return risks[index];
}

// random shuffle of elements in the list
void shuffle_elements(std::vector<int>& v, int threadid = 0)
{
	int n = v.size();
    for (int i = 0; i < n - 1; i++)
    {
        int j = i + rand_multithread(threadid) % (n - i);
        std::swap(v[i], v[j]);
    }
}

// returns true if two numbers are 'close enough' - this is to not have numerical issues
bool equal(double a, double b)
{
	if(a < b - 1e-8 || a > b + 1e-8) return false;
	return true;
}


// Resource class, everything is constant here
class Resource {
 public:
	int id;       								// id, starts at 0
	string name;								// name
	vector<double> min_consumption_at_ts; 		// [t] is the minimum resource usage at time step t, time steps start from zero
	vector<double> max_consumption_at_ts; 		// [t] is the maximum resource usage at time step t, time steps start from zero
};

// Intervention class
class Intervention {
 public:
	int id;       								 // id, starts from 0
	string name;  								 // name
	int tmax;     								 // latest possible start
	vector<int> duration_if_start_at;            // [t] is the duration of intervention if it starts at time step t, time steps start from zero
	vector<int> required_resources;              // list of resources required for performing intervention
	vector<vector<vector<double> > > demandRTD;  // [r][t][D] is how much of resource r is consumed at time step t if intervention starts at t' = t - D
	vector<vector<vector<double> > > riskTDS;    // [t][D][s] is risk for scenario s at time step t if intervention starts at t' = t - D, we use D instead of t' in the list to have smaller size

	// those are some useful informations that help to speed up the algorithm - TODO maybe drop some
	vector<vector<bool> > has_same_risk_over_all_scenarios;
	bool is_constant;
	vector<bool> is_constant_at_ts;
	bool belongsToAnyExclusion;
	vector<vector<double> > risk_total_at_ts_if_start_at;
};

// Data class, whole problem instance is loaded here
class Data {
public:
	string datafilename;  							// path to file
	int T;                							// number of time steps, time steps start from zero
	double Q;             							// quantile
	double alpha;         							// alpha
	double beta;          							// 1 - alpha
	int nmbResources;     							// number of resources
	int nmbInterventions; 							// number of interventions
	vector<int> nmb_scenarios_for_ts;          		// [t] is the number of scenarios at time step t, time steps start from zero
	vector<Resource*> resources_list;          		// list of resources
	vector<Intervention*> interventions_list;  		// list of interventions
	vector<vector<int> > season_ts;     			// [0] is list of time steps in FULL season, [1] in WINTER, [2] in SUMMER, [3] in INTER
	vector<vector<int> > exclusions;    			// list of exclusions, triplets (i1, i2, season);
	int min_nmb_scenarios_at_ts = 1000000;
	int max_nmb_scenarios_at_ts = 0;

	// [i][j] is the mean cost if intervention i starts at time step j
	// it simplifies mean cost calculation and can be pre-calculated
	vector<vector<double> > Cost_i_j;

 public:
	void print_basic()
	{
		cout << "------------------- bench -------------------------------" << endl;
		cout << datafilename << endl;
		cout << "   # time steps " << T << endl;
		cout << "# interventions " << nmbInterventions << endl;
		cout << "    # resources " << nmbResources << endl;
		cout << "       quantile " << Q << endl;
		cout << "          alpha " << alpha << endl;
		cout << "    # scenarios " << min_nmb_scenarios_at_ts << "-" << max_nmb_scenarios_at_ts << endl;
	    cout << "   # exclusions " << exclusions.size() << endl;
	    cout << "---------------------------------------------------------" << endl;
	}
	void precalculate_stuff()
	{
		Cost_i_j.resize(nmbInterventions, vector<double>(T, 0));
		for(unsigned int i = 0; i < nmbInterventions; i++)
		{
			Intervention* I = interventions_list[i];
			for(int S = 0; S < T; S++)
			{
				int E = S + I->duration_if_start_at[S];
				double risk = 0.0;
				if(E > T) continue;
				for(int t = S; t < E && t < T; t++)
				{
					for(int sc = 0; sc < nmb_scenarios_for_ts[t]; sc++)
					{
						if(I->riskTDS[t].size() > t - S && I->riskTDS[t][t - S].size() > sc) risk += (I->riskTDS[t][t - S][sc] / nmb_scenarios_for_ts[t]);
					}
				}
				risk = risk / T;
				Cost_i_j[i][S] = risk;
			}

			I->risk_total_at_ts_if_start_at.resize(T, vector<double>(T, 0));
			for(int t2 = 0; t2 < T; t2++)
			{
				for(int t1 = 0; t1 <= t2; t1++)
				{
					if(t2 - t1 >= I->riskTDS[t2].size()) continue;
					double risk = 0.0;
					for(int sc = 0; sc < nmb_scenarios_for_ts[t2]; sc++)
					{
						if(sc >= I->riskTDS[t2][t2 - t1].size()) continue;
					    risk += I->riskTDS[t2][t2 - t1][sc];
					}
					I->risk_total_at_ts_if_start_at[t2][t1] = risk /*/ nmb_scenarios_for_ts[t2]*/;
				}
			}
		}
	}
};

std::pair<int, int> calculate_overlap(Intervention* I1, int start1, Intervention* I2, int start2)
{
	std::pair<int, int> overlap;
	overlap.first = overlap.second = -1;
	int end1 = start1 + I1->duration_if_start_at[start1] - 1;
	int end2 = start2 + I2->duration_if_start_at[start2] - 1;
	if(start1 >= start2 && start1 <= end2) overlap.first = start1;
	if(start2 >= start1 && start2 <= end1) overlap.first = start2;
	if(end1 >= start2 && end1 <= end2) overlap.second = end1;
	if(end2 >= start1 && end2 <= end1) overlap.second = end2;
	return overlap;
}

class Solution {
public:
	Data* data;
	vector<int> start;  // solution vector: start time of each intervention
public:
	double curr_cost;
	vector<vector<double> > curr_res_consumption_at_ts;
	vector<double> curr_excess_at_ts;
	vector<double> currRiskAtT;
	vector<int> currNmbNonconstantAtT;
	vector<vector<double> > currRiskForScenariosAtT;
public:
	Solution(Data* d) {
		data = d;
		start.resize(d->nmbInterventions, 0);
		for(int i = 0; i < d->nmbInterventions; i++) start[i] = 0;
		curr_cost = 0.0;
		curr_res_consumption_at_ts.resize(d->nmbResources, vector<double>(d->T, 0));
		curr_excess_at_ts.resize(d->T, 0);
		currRiskAtT.resize(d->T, 0);
		currNmbNonconstantAtT.resize(d->T, 0);
		currRiskForScenariosAtT.resize(d->T);
		for(int t = 0; t < d->T; t++) currRiskForScenariosAtT[t].resize(d->nmb_scenarios_for_ts[t], 0);
	}
	Solution(Data* d, string solfilename) {
		data = d;
		start.resize(d->nmbInterventions, 0);
		curr_cost = 0.0;
		curr_res_consumption_at_ts.resize(d->nmbResources, vector<double>(d->T, 0));
		curr_excess_at_ts.resize(d->T, 0);
		currRiskAtT.resize(d->T, 0);
		currNmbNonconstantAtT.resize(d->T, 0);
		currRiskForScenariosAtT.resize(d->T);
		for(int t = 0; t < d->T; t++)
		{
			currRiskForScenariosAtT[t].resize(d->nmb_scenarios_for_ts[t], 0);
		}
		fstream file(solfilename.c_str(), ios_base::in);
		string name;
		int ts;
		for(int i = 0; i < d->nmbInterventions; i++)
		{
			file >> name >> ts;
			start[i] = ts - 1;
		}
		update_costs();
	}
	bool check(bool print = true);
	double calculate_mean_risk(bool print = false);
	double calculate_excess_obj(bool print = false);
	double calculate_cost() { return data->alpha * calculate_mean_risk() + data->beta * calculate_excess_obj(); }
	double recalculate_and_update_excess_at_ts(int t);
	double calculate_excess_at_ts_with_change(int t, int i, int old_start_i, int new_start_i);
	double calculate_excess_at_ts_with_double_change(int t, int i1, int old_start_i1, int new_start_i1, int i2, int old_start_i2, int new_start_i2);
	double calculate_excess_at_ts_with_drop(int t, int i , int oldStart);
	void update_costs()
	{
		curr_cost = calculate_cost();
		for(int t = 0; t < data->T; t++) curr_excess_at_ts[t] = recalculate_and_update_excess_at_ts(t);
		recalculate_resources_consumption();
	}

	void update_current_risks_at_timestep(int t, int i1, int old_start_i1, int new_start_i1, int i2, int old_start_i2, int new_start_i2)
	{
		Intervention* I1 = data->interventions_list[i1];
		Intervention* I2 = (i2 >= 0) ? data->interventions_list[i2] : 0;

		// if intervention I1 is dropped from t
		if(t >= old_start_i1 && t < old_start_i1 + I1->duration_if_start_at[old_start_i1])
		{
			int S = old_start_i1;
			if(I1->has_same_risk_over_all_scenarios[t][S] == false) currNmbNonconstantAtT[t]--;
			for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
			{
				if(I1->riskTDS[t].size() <= t - S || I1->riskTDS[t][t - S].size() <= sc) continue;
			    currRiskAtT[t] -= I1->riskTDS[t][t - S][sc];
			    currRiskForScenariosAtT[t][sc] -= I1->riskTDS[t][t - S][sc];
			}
		}
		// if intervention I2 is dropped from t
		if(I2 != 0 && t >= old_start_i2 && t < old_start_i2 + I2->duration_if_start_at[old_start_i2])
		{
			int S = old_start_i2;
			if(I2->has_same_risk_over_all_scenarios[t][S] == false) currNmbNonconstantAtT[t]--;
			for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
			{
				if(I2->riskTDS[t].size() <= t - S || I2->riskTDS[t][t - S].size() <= sc) continue;
			    currRiskAtT[t] -= I2->riskTDS[t][t - S][sc];
				currRiskForScenariosAtT[t][sc] -= I2->riskTDS[t][t - S][sc];
			}

		}
		// if intervention I1 is added to t
		if(t >= new_start_i1 && t < new_start_i1 + I1->duration_if_start_at[new_start_i1])
		{
			int S = new_start_i1;
			if(I1->has_same_risk_over_all_scenarios[t][S] == false) currNmbNonconstantAtT[t]++;
			for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
			{
				if(I1->riskTDS[t].size() <= t - S || I1->riskTDS[t][t - S].size() <= sc) continue;
			    currRiskAtT[t] += I1->riskTDS[t][t - S][sc];
				currRiskForScenariosAtT[t][sc] += I1->riskTDS[t][t - S][sc];
			}
		}
		// if intervention I2 is added to t
		if(I2 != 0 && t >= new_start_i2 && t < new_start_i2 + I2->duration_if_start_at[new_start_i2])
		{
			int S = new_start_i2;
			if(I2->has_same_risk_over_all_scenarios[t][S] == false) currNmbNonconstantAtT[t]++;
			for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
			{
				if(I2->riskTDS[t].size() <= t - S || I2->riskTDS[t][t - S].size() <= sc) continue;
			    currRiskAtT[t] += I2->riskTDS[t][t - S][sc];
				currRiskForScenariosAtT[t][sc] += I2->riskTDS[t][t - S][sc];
			}
		}

	}

	void update_costs_and_consumption_with_move(int i1, int old1, int new1, int i2 = -1, int old2 = -1, int new2 = -1)
	{
		set<int> affected_timesteps;
		for(int t = old1; t < old1 + data->interventions_list[i1]->duration_if_start_at[old1]; t++) affected_timesteps.insert(t);
		for(int t = new1; t < new1 + data->interventions_list[i1]->duration_if_start_at[new1]; t++) affected_timesteps.insert(t);
		if(i2 >= 0)
		{
			for(int t = old2; t < old2 + data->interventions_list[i2]->duration_if_start_at[old2]; t++) affected_timesteps.insert(t);
			for(int t = new2; t < new2 + data->interventions_list[i2]->duration_if_start_at[new2]; t++) affected_timesteps.insert(t);
		}

		// update total cost
		if(i2 < 0) curr_cost = calculate_cost_with_change(i1, old1, new1);
		else       curr_cost = calculate_cost_with_double_change(i1, old1, new1, i2, old2, new2);

		// update excess in each affected time step
		for(set<int>::iterator it = affected_timesteps.begin(); it != affected_timesteps.end(); it++)
		{
			int t = *it;
			if(i2 < 0) curr_excess_at_ts[t] = calculate_excess_at_ts_with_change(t, i1, old1, new1);
			else       curr_excess_at_ts[t] = calculate_excess_at_ts_with_double_change(t, i1, old1, new1, i2, old2, new2);

			update_current_risks_at_timestep(t, i1, old1, new1, i2, old2, new2);
		}

		recalculate_resources_consumption();
	}

	void write_to_file(string filename)
	{
		if(check())
		{
			fstream file(filename.c_str(), ios_base::out);
			for(int i = 0; i < start.size(); i++) file << data->interventions_list[i]->name << " " << start[i] + 1 << endl;
			file.close();
		}
	}

	bool check_with_change(int i, int new_start_i);
	bool check_with_double_change(int i1, int new_start_i1, int i2, int new_start_i2);
	double calculate_cost_with_change(int i, int old_start_i, int new_start_i);
	double calculate_cost_with_double_change(int i1, int old_start_i1, int new_start_i1, int i2, int old_start_i2, int new_start_i2);
	void local_search_rand(int timelimit = 36000, int maxLoops = 100000, int threadid = 0, bool print = false);
	void local_search_desc_pert(int timelimit = 36000, int maxLoops = 100000, int threadid = 0, bool print = false);
	void perturbate(int nmbPertMoves, int threadid, double tol = 10000);
	void recalculate_resources_consumption()
	{
		for(int r = 0; r < data->nmbResources; r++) for(int t = 0; t < data->T; t++) curr_res_consumption_at_ts[r][t] = 0;
		for(unsigned int i = 0; i < start.size(); i++)
		{
			Intervention* I = data->interventions_list[i];
			int S = start[i];
			int E = S + I->duration_if_start_at[S];
			for(int t = S; t < E; t++)
			{
				for(int &r : I->required_resources)
				{
					if(t - S >= I->demandRTD[r][t].size()) continue;
					curr_res_consumption_at_ts[r][t] += I->demandRTD[r][t][t - S];
				}
			}
		}
	}
};

bool Solution::check(bool print)
{
	if(start.size() != data->nmbInterventions) { cout << "start.size() != data->nmbInterventions\n"; return false; }
	for (int &v : start)
	{
		if(v < 0 || v >= data->T)
		{
			cout << "Non-performed interventions exist\n";
			return false;
		}
	}
	for(unsigned int i = 0; i < start.size(); i++)
	{
		int E = start[i] + data->interventions_list[i]->duration_if_start_at[start[i]];
		if(E > data->T)
		{
			if(print) cout << "IIntervention " << i << " ends outside of time horizon!\n";
			return false;
		}
	}
	// check resources consumption
	vector<vector<double> > res_consumption_at_ts_recalculated(data->nmbResources, vector<double>(data->T, 0));
	for(unsigned int i = 0; i < start.size(); i++)
	{
		Intervention* I = data->interventions_list[i];
		int S = start[i];
		int E = S + I->duration_if_start_at[S];
		for(int t = S; t < E; t++)
		{
			for(int &r : I->required_resources)
			{
				if(I->demandRTD[r][t].size() > t - S) res_consumption_at_ts_recalculated[r][t] += I->demandRTD[r][t][t - S];
			}
		}
	}
	bool ok = true;
	for(int r = 0; r < data->nmbResources; r++)
	{
		for(int t = 0; t < data->T; t++)
		{
			if(res_consumption_at_ts_recalculated[r][t] < data->resources_list[r]->min_consumption_at_ts[t] - 9e-6 ||
			   res_consumption_at_ts_recalculated[r][t] > data->resources_list[r]->max_consumption_at_ts[t] + 9e-6)
			{
				if(print)
				{
					cout << "Resource " << r << " consumption at ts " << t << " not ok : ";
					cout << res_consumption_at_ts_recalculated[r][t] << " ";
					cout << data->resources_list[r]->min_consumption_at_ts[t] << " ";
					cout << data->resources_list[r]->max_consumption_at_ts[t] << endl;
				}
				ok = false;
				if(print == false) return false;
			}

		}
	}
	// check exclusions
	for(unsigned int e = 0; e < data->exclusions.size(); e++)
	{
		int i1 = data->exclusions[e][0];
		int i2 = data->exclusions[e][1];
		int season = data->exclusions[e][2];
		std::pair<int, int> overlap = calculate_overlap(data->interventions_list[i1], start[i1], data->interventions_list[i2], start[i2]);
		int overlap_start = overlap.first;
		int overlap_end = overlap.second;
		for(int &t : data->season_ts[season])
		{
			if(t >= overlap_start && t <= overlap_end)
			{
				if(print)  cout << "Exclusion not respected: interventions " << i1 << " and " << i2 << " at time step " << t << endl;
				ok = false;
				if(print == false) return false;
			}
		}
	}
	return ok;
}

double Solution::calculate_mean_risk(bool print)
{
	double mean_risk = 0.0;
	for(int i = 0; i < data->nmbInterventions; i++)
	{
		mean_risk += data->Cost_i_j[i][start[i]];
	}
	return mean_risk;
}

double Solution::calculate_excess_obj(bool print)
{
	vector<double> risk_ts(data->T, 0);
	vector<vector<double> > risk_ts_scenario(data->T, vector<double>(0));
	for(int t = 0; t < data->T; t++)
	  for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
		risk_ts_scenario[t].push_back(0);
	for(unsigned int i = 0; i < start.size(); i++)
	{
		Intervention* I = data->interventions_list[i];
		int S = start[i];
		int E = S + I->duration_if_start_at[S];
		for(int t = S; t < E; t++)
		{
			for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
			{

			  if(I->riskTDS[t].size() <= t - S || I->riskTDS[t][t - S].size() <= sc) continue;

			  if(!isScenarioRemovedAtTS[t][sc]) risk_ts_scenario[t][sc] += I->riskTDS[t][t - S][sc];
			  risk_ts[t] += I->riskTDS[t][t - S][sc];
			}
		}
	}
	double total_excess = 0.0;
	for(int t = 0; t < data->T; t++)
	{
		if(risk_ts[t] == 0) continue;
		vector<double> sorted_list = risk_ts_scenario[t];
		std::sort(sorted_list.begin(), sorted_list.end());
		int index = ceil(data->Q * sorted_list.size()) - 1;
		double q = sorted_list[index];
		double mean_risk = risk_ts[t] / data->nmb_scenarios_for_ts[t];
		double ts_excess = q - mean_risk;
		if(ts_excess < 0) ts_excess = 0.0;
		total_excess += ts_excess;
	}
	double obj = total_excess / data->T;
	return obj;
}

double Solution::recalculate_and_update_excess_at_ts(int t)
{
	double risk = 0;
	int nmbNonconstantInterventionsAtTs = 0;
	vector<double> risk_for_scenario(data->nmb_scenarios_for_ts[t], 0);
	for(unsigned int i = 0; i < start.size(); i++)
	{
		Intervention* I = data->interventions_list[i];
		int S = start[i];
		int E = S + I->duration_if_start_at[S];
		if(t >= S && t < E)
		{
			if(I->has_same_risk_over_all_scenarios[t][S] == false) nmbNonconstantInterventionsAtTs++;
			risk += I->risk_total_at_ts_if_start_at[t][S];
			for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
			{
			  if(I->riskTDS[t].size() <= t - S || I->riskTDS[t][t - S].size() <= x) continue;
			  risk_for_scenario[x] += I->riskTDS[t][t - S][x];
			}
		}
	}

	currRiskAtT[t] = risk;
	currNmbNonconstantAtT[t] = nmbNonconstantInterventionsAtTs;
	currRiskForScenariosAtT[t] = risk_for_scenario;

	if(risk == 0) return 0;
	if(nmbNonconstantInterventionsAtTs == 0) return 0;
	double mean_risk = risk / data->nmb_scenarios_for_ts[t];
	int index = ceil(data->Q * data->nmb_scenarios_for_ts[t]) - 1;
	double q = select(risk_for_scenario, index);
	double excess = q - mean_risk;
	if(excess < 0) excess = 0.0;
	return excess;
}

double Solution::calculate_excess_at_ts_with_change(int t, int i, int old_start_i, int new_start_i)
{
	double risk = currRiskAtT[t];
	int nmbNonconstantInterventionsAtTs = currNmbNonconstantAtT[t];
	vector<double> risk_for_scenario = currRiskForScenariosAtT[t];

	Intervention* I = data->interventions_list[i];
	bool isdropped = (t >= old_start_i && t < old_start_i + I->duration_if_start_at[old_start_i]);
	bool isadded   = (t >= new_start_i && t < new_start_i + I->duration_if_start_at[new_start_i]);

	if(isdropped && I->has_same_risk_over_all_scenarios[t][old_start_i] == false) nmbNonconstantInterventionsAtTs--;
	if(isadded   && I->has_same_risk_over_all_scenarios[t][new_start_i] == false) nmbNonconstantInterventionsAtTs++;
	if(nmbNonconstantInterventionsAtTs == 0) return 0;


	if(isdropped) //if intervention is dropped from t
	{
		risk -= I->risk_total_at_ts_if_start_at[t][old_start_i];
		for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
		{
			if(I->riskTDS[t].size() > t - old_start_i && I->riskTDS[t][t - old_start_i].size() > x)
				risk_for_scenario[x] -= I->riskTDS[t][t - old_start_i][x];
		}
	}
	if(isadded) //if intervention is added to t
	{
		risk += I->risk_total_at_ts_if_start_at[t][new_start_i];
		for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
		{
			if(I->riskTDS[t].size() > t - new_start_i && I->riskTDS[t][t - new_start_i].size() > x)
		      risk_for_scenario[x] += I->riskTDS[t][t - new_start_i][x];
		}
	}

	if(risk == 0) return 0;
	double mean_risk = risk / data->nmb_scenarios_for_ts[t];
	int index  = ceil(data->Q * data->nmb_scenarios_for_ts[t]) - 1;
	double q = select(risk_for_scenario, index);
	double excess = q - mean_risk;
	if(excess < 0) excess = 0.0;
	return excess;
}

double Solution::calculate_excess_at_ts_with_double_change(int t, int i1, int old_start_i1, int new_start_i1, int i2, int old_start_i2, int new_start_i2)
{
	double risk = currRiskAtT[t];
	int nmbNonconstantInterventionsAtTs = currNmbNonconstantAtT[t];
	vector<double> risk_for_scenario = currRiskForScenariosAtT[t];

	Intervention* I1 = data->interventions_list[i1];
	Intervention* I2 = data->interventions_list[i2];

	// if intervention I1 is dropped from t
	if(t >= old_start_i1 && t < old_start_i1 + I1->duration_if_start_at[old_start_i1])
	{
		int S = old_start_i1;
		if(I1->has_same_risk_over_all_scenarios[t][S] == false) nmbNonconstantInterventionsAtTs--;
		risk -= I1->risk_total_at_ts_if_start_at[t][S];
		for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
		{
			if(I1->riskTDS[t].size() > t - S && I1->riskTDS[t][t - S].size() > x)
		      risk_for_scenario[x] -= I1->riskTDS[t][t - S][x];
		}
	}
	// if intervention I2 is dropped from t
	if(t >= old_start_i2 && t < old_start_i2 + I2->duration_if_start_at[old_start_i2])
	{
		int S = old_start_i2;
		if(I2->has_same_risk_over_all_scenarios[t][S] == false) nmbNonconstantInterventionsAtTs--;
		risk -= I2->risk_total_at_ts_if_start_at[t][S];
		for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
		{
			if(I2->riskTDS[t].size() > t - S && I2->riskTDS[t][t - S].size() > x)
		       risk_for_scenario[x] -= I2->riskTDS[t][t - S][x];
		}
	}
	// if intervention I1 is added to t
	if(t >= new_start_i1 && t < new_start_i1 + I1->duration_if_start_at[new_start_i1])
	{
		int S = new_start_i1;
		if(I1->has_same_risk_over_all_scenarios[t][S] == false) nmbNonconstantInterventionsAtTs++;
		risk += I1->risk_total_at_ts_if_start_at[t][S];
		for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
		{
			if(I1->riskTDS[t].size() > t - S && I1->riskTDS[t][t - S].size() > x)
				risk_for_scenario[x] += I1->riskTDS[t][t - S][x];
		}
	}
	// if intervention I2 is added to t
	if(t >= new_start_i2 && t < new_start_i2 + I2->duration_if_start_at[new_start_i2])
	{
		int S = new_start_i2;
		if(I2->has_same_risk_over_all_scenarios[t][S] == false) nmbNonconstantInterventionsAtTs++;
		risk += I2->risk_total_at_ts_if_start_at[t][S];
		for(int x = 0; x < data->nmb_scenarios_for_ts[t]; x++)
		{
			if(I2->riskTDS[t].size() > t - S && I2->riskTDS[t][t - S].size() > x)
				risk_for_scenario[x] += I2->riskTDS[t][t - S][x];
		}
	}


	if(risk == 0) return 0;
	if(nmbNonconstantInterventionsAtTs == 0) return 0;

	double mean_risk = risk / data->nmb_scenarios_for_ts[t];
	int index  = ceil(data->Q * data->nmb_scenarios_for_ts[t]) - 1;
	double q = select(risk_for_scenario, index);
	double excess = q - mean_risk;
	if(excess < 0) excess = 0.0;
	return excess;
}

bool Solution::check_with_change(int i, int new_start_i)
{
	Intervention* I = data->interventions_list[i];
	int old_start_i = start[i];
	// horizon
	int E = new_start_i + I->duration_if_start_at[new_start_i];
	if(E > data->T) {
		return false;
	}
	//resources consumption max
	for(int t = new_start_i; t < E; t++)
	{
		for(int &r : I->required_resources)
		{
			double RC = curr_res_consumption_at_ts[r][t];
			if(I->demandRTD[r][t].size() > t - new_start_i) RC +=  I->demandRTD[r][t][t - new_start_i];
			if(I->demandRTD[r][t].size() > t - old_start_i) RC -=  I->demandRTD[r][t][t - old_start_i];
			if(RC > data->resources_list[r]->max_consumption_at_ts[t] + 9e-6) return false;
			if(RC < data->resources_list[r]->min_consumption_at_ts[t] - 9e-6) return false;
		}
	}

	//resources consumption min
	for(int t = old_start_i; t < old_start_i + I->duration_if_start_at[old_start_i]; t++)
	{
		for(int &r : I->required_resources)
		{
			double RC = curr_res_consumption_at_ts[r][t];
			if(I->demandRTD[r][t].size() > t - old_start_i) RC -=  I->demandRTD[r][t][t - old_start_i];
			if(I->demandRTD[r][t].size() > t - new_start_i) RC +=  I->demandRTD[r][t][t - new_start_i];
			if(RC > data->resources_list[r]->max_consumption_at_ts[t] + 9e-6) return false;
			if(RC < data->resources_list[r]->min_consumption_at_ts[t] - 9e-6) return false;
		}
	}

	// check exclusions
	if(I->belongsToAnyExclusion)
	{
		for(int e = 0; e < data->exclusions.size(); e++)
		{
			int i1 = data->exclusions[e][0];
			int i2 = data->exclusions[e][1];
			int season = data->exclusions[e][2];
			if(i != i1 && i != i2) continue;
			std::pair<int, int> overlap;
			if(i == i1) overlap = calculate_overlap(data->interventions_list[i1], new_start_i, data->interventions_list[i2], start[i2]);
			if(i == i2) overlap = calculate_overlap(data->interventions_list[i1], start[i1], data->interventions_list[i2], new_start_i);
			int overlap_start = overlap.first;
			int overlap_end = overlap.second;
			for(int &t : data->season_ts[season])
			{
				if(t >= overlap_start && t <= overlap_end) {
					return false;
				}
			}
		}
	}
	return true;
}

double Solution::calculate_cost_with_change(int i, int old_start_i, int new_start_i)
{
	double cost = curr_cost;
	// mean risk difference
	double mean_risk_diff = data->alpha * data->Cost_i_j[i][new_start_i] -
			                data->alpha * data->Cost_i_j[i][old_start_i];
	cost += mean_risk_diff;
	if(data->interventions_list[i]->is_constant) return cost;

	// excess difference
	set<int> affected_timesteps;
	for(int t = old_start_i; t < old_start_i + data->interventions_list[i]->duration_if_start_at[old_start_i]; t++)
		affected_timesteps.insert(t);
	for(int t = new_start_i; t < new_start_i + data->interventions_list[i]->duration_if_start_at[new_start_i]; t++)
		affected_timesteps.insert(t);

	for(set<int>::iterator it = affected_timesteps.begin(); it != affected_timesteps.end(); it++)
	{
		int t = *it;

		if(equal(data->interventions_list[i]->risk_total_at_ts_if_start_at[t][old_start_i],
		         data->interventions_list[i]->risk_total_at_ts_if_start_at[t][new_start_i])) continue;  // TODO - is this correct???

		cost -= ((data->beta) * curr_excess_at_ts[t] * 1.0 / data->T);
		cost += ((data->beta) * calculate_excess_at_ts_with_change(t, i, old_start_i, new_start_i) * 1.0 / data->T);
	}

	return cost;
}

bool Solution::check_with_double_change(int i1, int new_start_i1, int i2, int new_start_i2)
{
	Intervention* I1 = data->interventions_list[i1];
	Intervention* I2 = data->interventions_list[i2];
	int old_start_i1 = start[i1];
	int old_start_i2 = start[i2];

	// horizon
	int E1 = new_start_i1 + I1->duration_if_start_at[new_start_i1];
	int E2 = new_start_i2 + I2->duration_if_start_at[new_start_i2];
	if(E1 > data->T || E2 > data->T) return false;

	// resources consumption
	set<int> affected_timesteps;
	for(int t = start[i1]; t < start[i1] + data->interventions_list[i1]->duration_if_start_at[start[i1]]; t++)
		affected_timesteps.insert(t);
	for(int t = start[i2]; t < start[i2] + data->interventions_list[i2]->duration_if_start_at[start[i2]]; t++)
		affected_timesteps.insert(t);
	for(int t = new_start_i1; t < new_start_i1 + data->interventions_list[i1]->duration_if_start_at[new_start_i1]; t++)
		affected_timesteps.insert(t);
	for(int t = new_start_i2; t < new_start_i2 + data->interventions_list[i2]->duration_if_start_at[new_start_i2]; t++)
		affected_timesteps.insert(t);

	for(set<int>::iterator it = affected_timesteps.begin(); it != affected_timesteps.end(); it++)
	{
		int t = *it;
		for(int r = 0; r < data->nmbResources; r++)
		{
			double RC = curr_res_consumption_at_ts[r][t];
			if(I1->demandRTD[r][t].size() > t - new_start_i1) RC +=  I1->demandRTD[r][t][t - new_start_i1];
			if(I1->demandRTD[r][t].size() > t - old_start_i1) RC -=  I1->demandRTD[r][t][t - old_start_i1];
			if(I2->demandRTD[r][t].size() > t - new_start_i2) RC +=  I2->demandRTD[r][t][t - new_start_i2];
			if(I2->demandRTD[r][t].size() > t - old_start_i2) RC -=  I2->demandRTD[r][t][t - old_start_i2];
			if(RC < data->resources_list[r]->min_consumption_at_ts[t] - 9e-6) return false;
			if(RC > data->resources_list[r]->max_consumption_at_ts[t] + 9e-6) return false;
		}
	}
	// check exclusions
	if(I1->belongsToAnyExclusion || I2->belongsToAnyExclusion)
	{
		for(int e = 0; e < data->exclusions.size(); e++)
		{
			int i1e = data->exclusions[e][0];
			int i2e = data->exclusions[e][1];
			int season = data->exclusions[e][2];
			if(i1 != i1e && i1 != i2e && i2 != i1e && i2 != i2e) continue;
			int start1e = start[i1e];
			int start2e = start[i2e];
			if(i1e == i1) start1e = new_start_i1;
			if(i1e == i2) start1e = new_start_i2;
			if(i2e == i1) start2e = new_start_i1;
			if(i2e == i2) start2e = new_start_i2;
			std::pair<int, int> overlap = calculate_overlap(data->interventions_list[i1e], start1e, data->interventions_list[i2e], start2e);
			int overlap_start = overlap.first;
			int overlap_end = overlap.second;
			for(int &t : data->season_ts[season])
			{
				if(t >= overlap_start && t <= overlap_end) {
					return false;
				}
			}
		}
	}
	return true;
}

double Solution::calculate_cost_with_double_change(int i1, int old_start_i1, int new_start_i1, int i2, int old_start_i2, int new_start_i2)
{
	double cost = curr_cost;

	// mean risk difference
	double mean_risk_diff = 0;
	mean_risk_diff -= data->alpha * data->Cost_i_j[i1][old_start_i1];
	mean_risk_diff += data->alpha * data->Cost_i_j[i1][new_start_i1];
	mean_risk_diff -= data->alpha * data->Cost_i_j[i2][old_start_i2];
	mean_risk_diff += data->alpha * data->Cost_i_j[i2][new_start_i2];
	cost += mean_risk_diff;
	if(data->interventions_list[i1]->is_constant &&
	   data->interventions_list[i2]->is_constant) return cost;

	// excess difference
	set<int> affected_timesteps;
	for(int t = old_start_i1; t < old_start_i1 + data->interventions_list[i1]->duration_if_start_at[old_start_i1]; t++)
		affected_timesteps.insert(t);
	for(int t = new_start_i1; t < new_start_i1 + data->interventions_list[i1]->duration_if_start_at[new_start_i1]; t++)
		affected_timesteps.insert(t);
	for(int t = old_start_i2; t < old_start_i2 + data->interventions_list[i2]->duration_if_start_at[old_start_i2]; t++)
		affected_timesteps.insert(t);
	for(int t = new_start_i2; t < new_start_i2 + data->interventions_list[i2]->duration_if_start_at[new_start_i2]; t++)
		affected_timesteps.insert(t);

	for(set<int>::iterator it = affected_timesteps.begin(); it != affected_timesteps.end(); it++)
	{
		int t = *it;
		cost -= ((data->beta) * curr_excess_at_ts[t] * 1.0 / data->T);
		cost += ((data->beta) *
			calculate_excess_at_ts_with_double_change(t, i1, old_start_i1, new_start_i1, i2, old_start_i2, new_start_i2) * 1.0 / data->T);
	}

	return cost;
}

int calc_pos(Data* data, Solution* sol, int i, int S)
{
	int pos = 1;
	for(int t = 0; t <= data->interventions_list[i]->tmax; t++)
		if(data->Cost_i_j[i][t] < data->Cost_i_j[i][S])
			pos++;
	return pos;
}

void Solution::local_search_desc_pert(int timelimit, int maxLoops, int threadid, bool print)
{
	update_costs();
	long int ls_start_time = time(0);
	double bestOfAll = calculate_cost();
	vector<int> bestOfAllvector = start;
	vector<int> interventions_indices;
	for(int i = 0; i < data->nmbInterventions; i++) interventions_indices.push_back(i);
	int loop = 0;
	while(loop++ < maxLoops)
	{
		if(time(0) - ls_start_time >= timelimit) break;
		if(time(0) - programStartTime >= timeLimitInSeconds) break;

		// descent
		bool thereIsImprovement = true;
		update_costs();
		shuffle_elements(interventions_indices, threadid);
		while(thereIsImprovement)
		{
			shuffle_elements(interventions_indices, threadid);
			if(time(0) - ls_start_time >= timelimit) break;
			thereIsImprovement = false;

			for(int ind = 0; ind < data->nmbInterventions && ind < 75; ind++)
			{
				if(time(0) - ls_start_time >= timelimit) break;
				if(time(0) - programStartTime >= timeLimitInSeconds) break;
			    int i = interventions_indices[ind];
				for(int t = 0; t <= data->interventions_list[i]->tmax; t++)
				{
					  if(start[i] == t) continue;
					  if(check_with_change(i, t) == false) continue;
					  double newcost = calculate_cost_with_change(i, start[i], t);
					  if(newcost < curr_cost - 1e-8)
					  {
						  int oldstart = start[i];
						  start[i] = t;
						  update_costs_and_consumption_with_move(i, oldstart, t);
						  thereIsImprovement = true;
					  }
				}
			}

		}

		if(curr_cost < bestOfAll) { bestOfAll = curr_cost; bestOfAllvector = start; }

		// perturbation
		int nmbPertMoves = 3;
		int nmbMoves = 0;
		double worstcost = curr_cost + 3;
		int nmbTries = 0;
		while(nmbMoves < nmbPertMoves)
		{
			nmbTries++;
			if(nmbTries >= 10000) break;
			int ii = rand_multithread(threadid) % data->nmbInterventions;
			int new_start = rand_multithread(threadid) % (data->interventions_list[ii]->tmax + 1);
			if(new_start != start[ii] && check_with_change(ii, new_start) &&
					calculate_cost_with_change(ii, start[ii], new_start) <= worstcost)
			{
				int oldstart = start[ii];
				start[ii] = new_start;
				update_costs_and_consumption_with_move(ii, oldstart, new_start);
				nmbMoves++;
			}
			worstcost += 1;
		}

		if(curr_cost < bestOfAll - 1e-8) { bestOfAll = curr_cost; bestOfAllvector = start; }
		// reset to best
		if(loop % 2 == 0) { start = bestOfAllvector; update_costs(); }
	}
	start = bestOfAllvector;
	update_costs();
}

void Solution::local_search_rand(int timelimit, int maxLoops, int threadid, bool print)
{
	update_costs();
	long int ls_start_time = time(0);
	int iter = 0;
	int maxIter = 33000;
	double bestcost = calculate_cost();
	double bestOfAll = calculate_cost();
	vector<int> bestvector = start;

	for(int loop = 0; loop < maxLoops; loop++)
	{
		int R = 500 + rand_multithread(threadid) % 5000;
		if(time(0) - ls_start_time >= timelimit) break;

		if(time(0) - programStartTime > timeLimitInSeconds) break;

		// shift
		iter = 0;
		while(iter++ < maxIter)
		{
			if(time(0) - ls_start_time >= timelimit) break;
			if(time(0) - programStartTime > timeLimitInSeconds) break;
			int ii = rand_multithread(threadid) % data->nmbInterventions;
			int new_start = rand_multithread(threadid) % (data->interventions_list[ii]->tmax + 1);
			if(new_start == start[ii]) continue;

			double posOld = calc_pos(data, this, ii, start[ii]);
			double posNew = calc_pos(data, this, ii, new_start);
			if(abs(posOld - posNew) > 10 && rand_multithread() % 100 < 50) continue;

			if(check_with_change(ii, new_start) == true)
			{
				bool acceptBad = (rand_multithread(threadid) % R < 3);
				double newcost = calculate_cost_with_change(ii, start[ii], new_start);
				if(newcost < bestcost - 1e-8 || (newcost < bestcost + 1000 && newcost < 1e15 && acceptBad))
				{
					int oldstart = start[ii];
					start[ii] = new_start;
					update_costs_and_consumption_with_move(ii, oldstart, new_start);
					if(newcost < bestOfAll - 1e-8) { bestOfAll = newcost; bestvector = start; }
					bestcost = newcost;
				}
			}
			// reset to best
			if(iter % 10000 == 0)
			{
				start = bestvector;
				update_costs();
			}
		}
		start = bestvector;
		update_costs();
		bestcost = curr_cost;

		// double shift
		iter = 0;
		while(iter++ < maxIter / 5)
		{
			if(time(0) - ls_start_time >= timelimit) break;
			if(time(0) - programStartTime >= timeLimitInSeconds) break;
			int ii1 = rand_multithread(threadid) % data->nmbInterventions;
			int ii2 = rand_multithread(threadid) % data->nmbInterventions;
			if(ii1 == ii2) continue;
			int new_start1 = rand_multithread(threadid) % (data->interventions_list[ii1]->tmax + 1);
			int new_start2 = rand_multithread(threadid) % (data->interventions_list[ii2]->tmax + 1);
			if(rand_multithread(threadid) % 100 < 80)
			{
				int newStart2UB = start[ii1] + data->interventions_list[ii1]->duration_if_start_at[start[ii1]] - 1;
				int newStart2LB = start[ii1] - data->interventions_list[ii2]->duration_if_start_at[start[ii1]] + 1;
				if(newStart2UB > data->interventions_list[ii2]->tmax) newStart2UB = data->interventions_list[ii2]->tmax;
				if(newStart2LB < 0) newStart2LB = 0;
				if(newStart2LB <= newStart2UB)
				{
					new_start2 = newStart2LB + rand_multithread(threadid) % (newStart2UB - newStart2LB + 1);
				}
			}

			if(new_start1 == start[ii1]) continue;
			if(new_start2 == start[ii2]) continue;

			if(check_with_double_change(ii1, new_start1, ii2, new_start2) == true)
			{
				bool acceptBad = (rand_multithread(threadid) % R < 2);
				double newcost = calculate_cost_with_double_change(ii1, start[ii1], new_start1, ii2, start[ii2], new_start2);

				if(newcost < bestcost - 1e-8 || (newcost < 1e15 && acceptBad))
				{
					int oldstart1 = start[ii1];
					int oldstart2 = start[ii2];
					start[ii1] = new_start1;
					start[ii2] = new_start2;
					update_costs_and_consumption_with_move(ii1, oldstart1, new_start1, ii2, oldstart2, new_start2);
					if(newcost < bestOfAll - 1e-8) { bestOfAll = newcost; bestvector = start; }
					bestcost = newcost;
				}
			}
			// reset to best
			if(iter % 10000 == 0)
			{
				start = bestvector;
				update_costs();
			}
		}

		start = bestvector;
		update_costs();
		bestcost = curr_cost;

		// swap
		iter = 0;
		while(iter++ < maxIter / 5)
		{
			if(time(0) - ls_start_time >= timelimit) break;
			if(time(0) - programStartTime >= timeLimitInSeconds) break;
			int ii1 = rand_multithread(threadid) % data->nmbInterventions;
			int ii2 = rand_multithread(threadid) % data->nmbInterventions;
			int new_start1 = start[ii2];
			int new_start2 = start[ii1];
			if(ii1 == ii2 || new_start1 == new_start2) continue;

			if(check_with_double_change(ii1, new_start1, ii2, new_start2) == true)
			{
				bool acceptBad = (rand_multithread(threadid) % R < 3);
				double newcost = calculate_cost_with_double_change(ii1, start[ii1], new_start1, ii2, start[ii2], new_start2);

				if(newcost < bestcost - 1e-8 || (newcost < 1e15 && acceptBad))
				{
					int oldstart1 = start[ii1];
					int oldstart2 = start[ii2];
					start[ii1] = new_start1;
					start[ii2] = new_start2;
					update_costs_and_consumption_with_move(ii1, oldstart1, new_start1, ii2, oldstart2, new_start2);
					if(newcost < bestOfAll - 1e-8) { bestOfAll = newcost; bestvector = start; }
					bestcost = newcost;
				}
			}
			// reset to best
			if(iter % 10000 == 0)
			{
				start = bestvector;
				update_costs();
			}
		}
		start = bestvector;
		update_costs();
	}
	start = bestvector;
	update_costs();
}

void Solution::perturbate(int nmbPertMoves, int threadid, double tol)
{
	update_costs();
	int nmbMoves = 0;
	double worstcost = curr_cost + tol;
	int nmbTries = 0;
	while(nmbMoves < nmbPertMoves)
	{
		nmbTries++;
		if(nmbTries >= 100000) break;
		int ii = rand_multithread(threadid) % data->nmbInterventions;
		int new_start = rand_multithread(threadid) % (data->interventions_list[ii]->tmax + 1);
		if(new_start != start[ii] && check_with_change(ii, new_start) &&
				calculate_cost_with_change(ii, start[ii], new_start) <= worstcost)
		{
			start[ii] = new_start;
			update_costs();
			nmbMoves++;
		}
		worstcost ++;
	}
}

void perturbatePlusLsLoop(Data* data, Solution* sol, int threadid = 0, int maxLoops = 50000, int timeLimit = 360000,
		int maxLoopsLSrand = 100000, int maxLoopsLSpert = 10000)
{
	vector<int> bestvector = sol->start;
	double bestcost = sol->calculate_cost();
	long int startTime = time(0);
	for(int i = 0; i < maxLoops; i++)
	{
		if(time(0) - programStartTime > timeLimitInSeconds) break;
		if(time(0) - startTime > timeLimit) break;
		if(i % 2 >= 0)
		{
			sol->start = bestvector;
			sol->update_costs();
		}
		sol->perturbate(1 + rand_multithread(threadid) % 3,  threadid);
		for(int i = 0; i < 2; i++)
		{
			sol->local_search_desc_pert(timeLimit / 4, maxLoopsLSpert, threadid);
			sol->local_search_rand(timeLimit / 4, maxLoopsLSrand, threadid);
		}
		sol->update_costs();
		if(sol->curr_cost < bestcost - 1e-8)
		{
			bestcost = sol->curr_cost;
			bestvector = sol->start;
		}
	}
	sol->start = bestvector;
	sol->update_costs();
}

Data* read_data(string filename);
double MIP_Q1(Data* data, Solution* sol, int MIP_TIME_LIMIT = 3600, bool startFromInitial = false);
pair<double, double> MIP_mean(Data* data, Solution* sol, int MIP_TIME_LIMIT = 36000, double gap = 0.00, bool NO_OBJECTIVE = false);

struct thread_data {
   int threadid;
   Data* data;
   Solution* sol;
   int nmbLoopsForImproveLS;
   int timeLimitLSrand;
   int timeLimitLSpert;
   int maxLoopsLSrand;
   int maxLoopsLSpert;
   int timeLimitPLS;
   int maxLoopsPLS;
   int maxLoopsPLSrand;
   int maxLoopsPLSpert;
};

void* perturbatePlusLS(void *threadarg)
{
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    perturbatePlusLsLoop(my_data->data, my_data->sol, my_data->threadid, my_data->maxLoopsPLS, my_data->timeLimitPLS, my_data->maxLoopsLSrand,
    		my_data->maxLoopsLSpert);
	pthread_exit(NULL);
}

void PLS(Data* data, Solution* sol, int maxLoopsLSrand, int maxLoopsLSpert, int plsLoops, int maxLoopsPLSInside)
{
   cout << "------------------- PLS -----------------------" << endl;
   pthread_t threads[NUM_THREADS];
   struct thread_data td[NUM_THREADS];
   Solution* thread_sols[NUM_THREADS];
   thread_sols[0] = sol;
   for(int i = 1; i < NUM_THREADS; i++)  thread_sols[i] = new Solution(data);
   for(int i = 0; i < NUM_THREADS; i++)
   {
	  td[i].threadid = i;
	  td[i].data = data;
	  td[i].sol = thread_sols[i];
	  td[i].timeLimitLSrand = 3600;
	  td[i].timeLimitLSpert = 3600;
	  td[i].maxLoopsLSrand = maxLoopsLSrand;
	  td[i].maxLoopsLSpert = maxLoopsLSpert;
   }

   // improve
   long int startTimePLS = time(0);
   for(int loop = 0; loop < plsLoops; loop++)
   {
	   if(time(0) - programStartTime > timeLimitInSeconds) break;
	   for(int i = 0; i < NUM_THREADS; i++)
	   {
		  td[i].sol->start = sol->start;
		  td[i].sol->update_costs();
		  td[i].maxLoopsPLS = maxLoopsPLSInside;
		  td[i].maxLoopsLSrand = maxLoopsLSrand;
		  td[i].maxLoopsLSpert = maxLoopsLSpert;
		  td[i].timeLimitPLS = 3600;
		  pthread_create(&threads[i], NULL, perturbatePlusLS, (void *)&td[i]);
	   }
	   for(int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);
	   // choose best solution
	   double bestCost = sol->calculate_cost();
	   for(int th = 0; th < NUM_THREADS; th++)
	   {
		   if(thread_sols[th]->calculate_cost() < bestCost - 1e-8)
		   {
			   bestCost = thread_sols[th]->calculate_cost();
			   sol->start = thread_sols[th]->start;
			   sol->update_costs();
		   }
	   }
	   cout << " PLS loop" << setw(3) << loop << setw(20) << bestCost << setw(10) << time(0) - programStartTime << endl;
	   sol->write_to_file(solutionfilename);
   }
  // cout << "PLS time " << time(0) - startTimePLS << endl;
}


// for updating coefficients in MIP objective
vector<vector<double> > penalscen;
vector<vector<double> > coefscen;
vector<double> coefobj;
vector<vector<int> > freq;

void updatePenalties(Data* data, Solution* sol, const int t)
{
	double obj1 = sol->currRiskAtT[t] / data->nmb_scenarios_for_ts[t];
	vector<double> risk_for_scenario = sol->currRiskForScenariosAtT[t];
	int index  = ceil(data->Q * data->nmb_scenarios_for_ts[t]) - 1;
	double q = select(risk_for_scenario, index);
	if(q > obj1)	{
		double vp = (q - obj1) / (data->nmb_scenarios_for_ts[t] - index);
		vp = vp * data->nmb_scenarios_for_ts[t];
		for(int s = 0; s < data->nmb_scenarios_for_ts[t]; s++)
		{
			if (sol->currRiskForScenariosAtT[t][s] > q)
				{
				penalscen[t][s] = penalscen[t][s] + vp;
				freq[t][s]++;
				}
		}
	}
}

double coefRisk(Data* data, const int interv,const int t0)
{
	double meanrisk = 0;
	int E = t0 + data->interventions_list[interv]->duration_if_start_at[t0];
	for(int t = t0; t < E && t < data->T; t++)
	{
		for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++)
		{
			if(data->interventions_list[interv]->riskTDS[t].size() > t - t0 &&
			   data->interventions_list[interv]->riskTDS[t][t - t0].size() > sc)
		      meanrisk += (data->interventions_list[interv]->riskTDS[t][t - t0][sc] * coefscen[t][sc]);
		}
	}
	return meanrisk;
}

void update_objective_coefficients(Data* data)
{
	for(int i = 0; i < data->T; i++)
	{
		coefscen[i][0] = 1.0;
		if (freq[i][0]) coefscen[i][0]+=(penalscen[i][0]/freq[i][0]);
		double sumcoef = coefscen[i][0];
		for(int s = 1; s < data->nmb_scenarios_for_ts[i]; s++)
		{
			coefscen[i][s] = 1.0;
			if (freq[i][s]) coefscen[i][s]+=(penalscen[i][s]/freq[i][s]);
			sumcoef += coefscen[i][s];
		}
		for(int s = 0; s < data->nmb_scenarios_for_ts[i]; s++)
			coefscen[i][s] = coefscen[i][s]/sumcoef;
	}
	int	v = 0;
	for(int i = 0; i < data->nmbInterventions; i++)
		for(int j = 0; j <= data->interventions_list[i]->tmax; j++)
		{
			coefobj[v] = coefRisk(data, i, j);
			v++;
		}
}

void* improvequick(void *threadarg)
{
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
	for(int k = 0; k < my_data->nmbLoopsForImproveLS; k++)
	{
	   if(time(0) - programStartTime > timeLimitInSeconds) break;
	   my_data->sol->local_search_rand(my_data->timeLimitLSrand, my_data->maxLoopsLSrand, my_data->threadid);
	   my_data->sol->local_search_desc_pert(my_data->timeLimitLSpert, my_data->maxLoopsLSpert, my_data->threadid);
	}

    perturbatePlusLsLoop(my_data->data, my_data->sol, my_data->threadid, my_data->maxLoopsPLS, my_data->timeLimitPLS, my_data->maxLoopsLSrand,
    		my_data->maxLoopsLSpert);

	pthread_exit(NULL);
}

void improve_quick_parallel(Data* data, Solution* sol)
{
   pthread_t threads[NUM_THREADS];
   struct thread_data td[NUM_THREADS];
   Solution* thread_sols[NUM_THREADS];
   thread_sols[0] = sol;
   for(int i = 1; i < NUM_THREADS; i++)  thread_sols[i] = new Solution(data);
   for(int i = 0; i < NUM_THREADS; i++)
   {
	  td[i].threadid = i;
	  td[i].data = data;
	  td[i].sol = thread_sols[i];
	  td[i].timeLimitLSrand = 1;
	  td[i].timeLimitLSpert = 1;
	  td[i].maxLoopsLSrand = 10000;
	  td[i].maxLoopsLSpert = 10000;
	  td[i].nmbLoopsForImproveLS = 2;
	  td[i].maxLoopsPLS = 100000;
	  td[i].timeLimitPLS = 4;
	  td[i].sol->start = sol->start;
	  td[i].sol->update_costs();
   }

   for(int i = 0; i < NUM_THREADS; i++)
   {
	  pthread_create(&threads[i], NULL, improvequick, (void *)&td[i]);
   }
   for(int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);
   // choose best solution
   double bestCost = sol->calculate_cost();
   for(int th = 0; th < NUM_THREADS; th++)
   {
	   double thCost = thread_sols[th]->calculate_cost();
	   if(thCost < bestCost - 1e-8)
	   {
		   bestCost = thCost;
		   sol->start = thread_sols[th]->start;
		   sol->update_costs();
	   }
   }
   sol->update_costs();
}

void removeScenariosAboveQ(Data* data, Solution* sol)
{
	cout << "removing scenarios..." << endl;
	for(int t = 0; t < data->T; t++)
	{
		vector<pair<double, int> > risk_scenario;
		for(int sc = 0; sc < data->nmb_scenarios_for_ts[t]; sc++) risk_scenario.push_back(pair<double, int>(0, sc));
		for(unsigned int i = 0; i < data->nmbInterventions; i++)
		{
			Intervention* I = data->interventions_list[i];
			int S = sol->start[i];
			if(S <= t && t < S + I->duration_if_start_at[S] && I->riskTDS[t].size() > t - S)
			{
				for(int sc = 0; sc < data->nmb_scenarios_for_ts[t] && sc < I->riskTDS[t][t - S].size(); sc++)
				{
					risk_scenario[sc].first += I->riskTDS[t][t - S][sc];
				}
			}
		}
		std::sort(risk_scenario.begin(), risk_scenario.end());
		int indexToRemoveFrom = ceil(data->Q * risk_scenario.size());
		vector<int> scenarios_to_remove;
		for(int i = indexToRemoveFrom; i < risk_scenario.size(); i++) scenarios_to_remove.push_back(risk_scenario[i].second);
		std::sort(scenarios_to_remove.begin(), scenarios_to_remove.end());
		for(int sci = scenarios_to_remove.size() - 1; sci >= 0; sci--)
		{
			int SC = scenarios_to_remove[sci];
			isScenarioRemovedAtTS[t][SC] = true;
		}
	}
}

Solution* removeScenariosAboveQAndSolveMIPQ1(Data* data, Solution* sol, int timeLimit = 3600)
{
	vector<int> solvector = sol->start;
	removeScenariosAboveQ(data, sol);
	double Qoriginal = data->Q;
	data->Q = 1;
	sol->update_costs();

	int remTime = timeLimitInSeconds - (time(0) - programStartTime);
	if(timeLimit > remTime) timeLimit = remTime;
	MIP_Q1(data, sol, timeLimit, true);


	// undo removing scenarios
	for(unsigned int x = 0; x < isScenarioRemovedAtTS.size(); x++)
		for(unsigned int y = 0; y < isScenarioRemovedAtTS[x].size(); y++)
			isScenarioRemovedAtTS[x][y] = false;

	data->Q = Qoriginal;
	sol->update_costs();
	return sol;
}

vector<int> solinitvector;
int feasibleSolutionTime;

void solve(Data* data, Solution* sol, int timeLimitForSolve)
{
	//****************** generate solutions by updating MIP objective coefficients *********************
	// the goal is to have coefficients such that optimal mean cost MIP solution is optimal/good
	// solution for original problem (mean + excess)
	vector<pair<double, vector<int> > > solution_pool;
	long int gspStartTime = time(0);
	double costmin = 1e15;

	int elapsedTime = time(0) - programStartTime;
	int remainingTime = timeLimitInSeconds - elapsedTime;

	int gspTimeLimit = (int) (4 * 60 * (timeLimitForSolve / 870.0));
	double gspTimeLimitMAX = (int) (remainingTime * 0.50);
	int mipTimeLimit = 30;
	if(feasibleSolutionTime > 25) mipTimeLimit = 45;
	int iter = 0;
	int itera = 0;
	while(true)
	{
		iter++;
		update_objective_coefficients(data);

		pair<double, double> res = MIP_mean(data, sol, mipTimeLimit, 0.00001);

		if(res.first < 1e15)
		{
			sol->update_costs();
			double objmip = res.first;
			double objbound = res.second;
			double gap = (objbound > 0) ? (objmip - objbound) / objbound : 1;
			if(gap > 0.01 * 1 / 100.0) {
				mipTimeLimit += 15;
				gspTimeLimit += 30;
				if(gspTimeLimit > gspTimeLimitMAX) gspTimeLimit = gspTimeLimitMAX;
			}

			itera++;

			double cost = sol->calculate_cost();
			if (cost < costmin)	{
				costmin = cost;
				sol->write_to_file(solutionfilename);
				itera=0;
			}
			cout << "iter "  << setw(4)  << iter
			     << " obj"   << setw(15) << objmip
				 << " LB"    << setw(15) << objbound
				 << " cost"  << setw(15) << setprecision(10) << cost
				 << " best"  << setw(15) << costmin
				 << " mipTL" << setw(5)  << mipTimeLimit
				 << " gspTL" << setw(5)  << gspTimeLimit
				 << " cpu"   << setw(4)  << time(0) - programStartTime
				 << endl;

			for(int t = 0; t < data->T; t++) updatePenalties(data, sol, t);
			solinitvector = sol->start;
			solution_pool.push_back(pair<double, vector<int> >(cost, sol->start));
		}
		else {
			cout << "MIP did not find solution! " << endl;
		}
		if(time(0) - gspStartTime > gspTimeLimit) break;

		if(itera > 2) // if no improvement in more than 2 iterations then decrease the penalties
		{
			//double red = 0.2 + ((rand_multithread() % 11) * 0.01);
			double red = 0.25;
			itera = 0;
			for(int t = 0; t < data->T; t++)
				for(int s = 0; s < data->nmb_scenarios_for_ts[t]; s++)
				{
					penalscen[t][s] = red * penalscen[t][s];
				}
		}
	}
	//**************************************************************************************************

	//************ take few best solutions obtained until now and do quick local search ****************
	// sometimes best solution is not the one that can be improved the most so we try few solutions
	// few restarts are made from each solution as well (since there is randomness in local search)
	// a single local search call takes 12 seconds, we will do no more than 11 restarts in total
	// so it will take around 2 minutes maximum
	sort(solution_pool.begin(), solution_pool.end());
	for(unsigned int i = 0; i < solution_pool.size() - 1; i++)
		if(equal(solution_pool[i].first, solution_pool[i + 1].first))
			solution_pool.erase(solution_pool.begin() + i--);
	for(unsigned int i = 0; i < solution_pool.size() && i < 8; i++)
		cout << setw(15) << setprecision(15) << solution_pool[i].first << endl;

	// few quick restarts
	vector<pair<double, vector<int> > > ls_pool;
	int nmbSolutionsToChooseFrom = 4;
	if(solution_pool.size() < 20) nmbSolutionsToChooseFrom = 3;
	if(solution_pool.size() < 10) nmbSolutionsToChooseFrom = 2;
	vector<int> nmbResetsForEachSolution = {3, 3, 3, 3, 2, 2, 2, 2};
	double bestAfterLS = 1e15;
	vector<int> best_solvector_after_ls;
	if(timeLimitForSolve > 25 * 60) nmbSolutionsToChooseFrom = 2 * nmbSolutionsToChooseFrom;
	for(unsigned int s = 0; s < nmbSolutionsToChooseFrom && s < solution_pool.size(); s++)
	{
		cout << "-----------------------------------------------" << endl;
		for(int r = 0; r < nmbResetsForEachSolution[s]; r++)
		{
			sol->start = solution_pool[s].second;
			sol->update_costs();
			cout << "start cost " << setw(20) << setprecision(15) << sol->curr_cost
				 << setw(10) << time(0) - programStartTime << endl;
			improve_quick_parallel(data, sol);
			cout << "   LS cost " << setw(20) << setprecision(15) << sol->curr_cost
				 << setw(10) << time(0) - programStartTime << endl;
			if(sol->curr_cost < bestAfterLS) {
				bestAfterLS = sol->curr_cost;
				best_solvector_after_ls = sol->start;
				sol->write_to_file(solutionfilename);
			}
			ls_pool.push_back(pair<double, vector<int> >(sol->curr_cost, sol->start));
		}
		cout << "-----------------------------------------------" << endl;
	}
	sol->start = best_solvector_after_ls;
	sol->update_costs();
	sol->write_to_file(solutionfilename);

	sort(ls_pool.begin(), ls_pool.end());
	for(unsigned int i = 0; i < ls_pool.size() - 1; i++)
		if(equal(ls_pool[i].first, ls_pool[i + 1].first)) ls_pool.erase(ls_pool.begin() + i--);
	for(unsigned int i = 0; i < ls_pool.size() && i < 3; i++)
		cout << setw(15) << setprecision(15) << ls_pool[i].first << endl;

	// few quick restarts again
	nmbSolutionsToChooseFrom = 3;
	if(ls_pool.size() < nmbSolutionsToChooseFrom) nmbSolutionsToChooseFrom = ls_pool.size();
	vector<int> nmbResetsForEachSolution2 = {2, 2, 1};
	bestAfterLS = 1e15;
	for(unsigned int s = 0; s < nmbSolutionsToChooseFrom && s < ls_pool.size(); s++)
	{
		for(int r = 0; r < nmbResetsForEachSolution2[s]; r++)
		{
			sol->start = ls_pool[s].second;
			sol->update_costs();
			cout << "start cost " << setw(20) << setprecision(15) << sol->curr_cost
				 << setw(10) << time(0) - programStartTime << endl;
			improve_quick_parallel(data, sol);
			cout << "   LS cost " << setw(20) << setprecision(15) << sol->curr_cost
				 << setw(10) << time(0) - programStartTime << endl;
			if(sol->curr_cost < bestAfterLS) {
				bestAfterLS = sol->curr_cost;
				best_solvector_after_ls = sol->start;
				sol->write_to_file(solutionfilename);
			}
			ls_pool.push_back(pair<double, vector<int> >(sol->curr_cost, sol->start));
		}
		cout << "-----------------------------------------------" << endl;
	}
	sol->start = best_solvector_after_ls;
	sol->update_costs();
	sol->write_to_file(solutionfilename);
	cout << "      BEST " << setw(20) << sol->curr_cost << endl;
	//**************************************************************************************************


	// now we will spend the remaining time to further optimize the best solution found until now
	// slower local search will be used
	// sometimes we will also use MIP with quantile = 1 to improve solution

	elapsedTime = time(0) - programStartTime;
	remainingTime = timeLimitInSeconds - elapsedTime;

	int mipQ1TimeLimit = (timeLimitForSolve > 30 * 60) ? 3 * 60 : 2 * 60;
	int q1loops = (timeLimitForSolve > 30 * 60) ? 2 : 1;
	int timeForQ1OrSlowerLS = (timeLimitForSolve > 30 * 60) ? 3 * 60 : 1 * 60;
	if(data->max_nmb_scenarios_at_ts < 100 && data->T * data->max_nmb_scenarios_at_ts < 100 * 100)
		timeForQ1OrSlowerLS += q1loops *  mipQ1TimeLimit;

	int timeLimitInSecondsOriginal = timeLimitInSeconds;
	if(remainingTime >= timeForQ1OrSlowerLS) timeLimitInSeconds -= timeForQ1OrSlowerLS;

	for(int i = 0; i < 10; i++)
	{
		if(time(0) - programStartTime > timeLimitInSeconds) break;
		sol->local_search_rand(1, 3);
		sol->local_search_desc_pert(1, 5);
		cout << setw(31) << sol->calculate_cost() << setw(10) << time(0) - programStartTime << endl;
		sol->write_to_file(solutionfilename);
	}
	PLS(data, sol, 1,  1,   20, 1);
	PLS(data, sol, 2,  2,   15, 2);
	improve_quick_parallel(data, sol);
	PLS(data, sol, 2,  2,   10, 1);
	PLS(data, sol, 2,  5,    5, 2);
	improve_quick_parallel(data, sol);
	PLS(data, sol, 2,  5,    5, 3);
	improve_quick_parallel(data, sol);
	PLS(data, sol, 2, 10,  10, 3);
	improve_quick_parallel(data, sol);
	PLS(data, sol, 2, 10,  10, 3);
	improve_quick_parallel(data, sol);
	PLS(data, sol, 2, 10, 1500, 2);

	timeLimitInSeconds = timeLimitInSecondsOriginal; // restore original time limit
	elapsedTime = time(0) - programStartTime;
	remainingTime = timeLimitInSeconds - elapsedTime;

	// Q1 loop
	if(remainingTime >= (q1loops *  mipQ1TimeLimit + 30)
			&& data->max_nmb_scenarios_at_ts < 100
			&& data->T * data->max_nmb_scenarios_at_ts < 100 * 100
	)
	{
		cout << "----------------- MIP_Q1 ----------------------" << endl;
		for(int l = 0; l < q1loops; l++)
		{
			cout << "  initial : " << setw(20) << setprecision(15) << sol->curr_cost << endl;
			sol = removeScenariosAboveQAndSolveMIPQ1(data, sol, mipQ1TimeLimit);
			cout << "after Q1 : " << setw(20) << setprecision(15) << sol->curr_cost << endl;
			improve_quick_parallel(data, sol);
			cout << "improved : " << setw(20) << setprecision(15) << sol->curr_cost << endl;
		}
	}

	// try to improve further until time limit is reached
	// this is slower local search
	PLS(data, sol, 2, 10, 1000, 5);

}


int main(int argc, char** argv)
{
	programStartTime = time(0);
	int seed = time(0) % 10000;
	int forcedSeed = -1;
	bool makereport = false;
	string datafilename = "";
	string initialsolutionfilename = "";
	// parse arguments
	for(int param = 1; param < argc; param++)
	{
		if(string(argv[param]) == string("-p"))          			datafilename = argv[param+1];
		else if(string(argv[param]) == string("-o"))     			solutionfilename = argv[param+1];
		else if(string(argv[param]) == string("-s"))     			seed = atoi(argv[param+1]);
		else if(string(argv[param]) == string("-t"))     			timeLimitInSeconds = atoi(argv[param+1]);
		else if(string(argv[param]) == string("-name"))        		{ cout << "S34" << endl; if(argc == 2) exit(0); }
		else if(string(argv[param]) == string("-forcedseed"))   	forcedSeed = atoi(argv[param+1]);
		else if(string(argv[param]) == string("-report"))   	    makereport = true;
		else if(string(argv[param]) == string("-i"))     			initialsolutionfilename = argv[param+1];
    }

	Data* data = read_data(datafilename);
	data->print_basic();
	Solution* sol;
	if(initialsolutionfilename == "") sol = new Solution(data);
	else sol = new Solution(data, initialsolutionfilename);
	timeLimitInSeconds -= 30;
	if(forcedSeed >= 0) seed = forcedSeed;
	for(int x = 0; x < NUM_THREADS; x++) srand_multithread(seed + x, x);
	cout << "reading time " << setw(7)  << time(0) - programStartTime << endl;

	//******************** first find any feasible solution (no objective in MIP) **********************
	long int feasibleSolutionStartTime = time(0);
	MIP_mean(data, sol, 36000, 0.00, true);
	feasibleSolutionTime = time(0) - feasibleSolutionStartTime;
	solinitvector = sol->start;
	cout << "solinit " << sol->calculate_cost() << setw(7)  << time(0) - programStartTime << endl;
	sol->write_to_file(solutionfilename);
	//**************************************************************************************************


	int nbvarij = 0;
	for (int i = 0; i < data->nmbInterventions;i++)  nbvarij += (1 + data->interventions_list[i]->tmax);
	penalscen.resize(data->T);
	coefscen.resize(data->T);
	freq.resize(data->T);
	for(int t = 0; t < data->T; t++) {
		penalscen[t].resize(data->nmb_scenarios_for_ts[t], 0.0);
		coefscen[t].resize(data->nmb_scenarios_for_ts[t], 0.0);
		freq[t].resize(data->nmb_scenarios_for_ts[t], 0);
	}
	coefobj.resize(nbvarij, 0.0);



	// if time limit is larger then we will repeat the solve procedure 3 times (1h30 = 3 * 30min)
	double bestObj = 1e15;
	vector<int> bestVector = sol->start;
	int nmbCallsTosolve = (timeLimitInSeconds > 89 * 60) ? 3 : 1;
	int timeLimitForSol = timeLimitInSeconds / nmbCallsTosolve;
	for(int x = 0; x < nmbCallsTosolve; x++)
	{
		timeLimitInSeconds = timeLimitForSol * (x + 1);
		solve(data, sol, timeLimitForSol);
		if(sol->calculate_cost() < bestObj)
		{
			bestObj = sol->calculate_cost();
			bestVector = sol->start;
			sol->write_to_file(solutionfilename);
		}
	}
	sol->start = bestVector;
	sol->update_costs();
	cout << "best solution " << sol->calculate_cost() << endl;

	cout << endl << endl;
	cout << "check " << sol->check(1) << endl;
	cout << "mean risk : " << setprecision(15) << sol->calculate_mean_risk() << endl;
	cout << "   excess : " << setprecision(15) << sol->calculate_excess_obj() << endl;
	cout << "    total : " << setprecision(15) << sol->calculate_cost() << endl;
	sol->write_to_file(solutionfilename);
	if(makereport)
	{
		fstream fileReport("report.csv", ios_base::out | ios_base::app);
		fileReport <<  datafilename << ";" <<  data->nmbInterventions << ";" << data->nmbResources << ";" <<  data->T
				   << ";" <<  data->min_nmb_scenarios_at_ts << ";" <<  data->max_nmb_scenarios_at_ts
				   << ";" <<  sol->calculate_mean_risk() << ";" <<  sol->calculate_excess_obj()
				   << ";" <<  seed
				   << ";" <<  time(0) - programStartTime
				   << ";" << setprecision(15) << sol->calculate_cost() << endl;
	}

	pthread_exit(NULL);
	return EXIT_SUCCESS;
}

// convert integer to string
string itos(int i) {stringstream s; s << i; return s.str(); }

double MIP_Q1(Data* data, Solution* sol, int MIP_TIME_LIMIT, bool startFromInitial)
{
	int nmbvar = 0;
	std::vector<std::vector<int> > key(data->nmbInterventions, std::vector<int>(data->T, -1));
	int K = 0;
	for (int i = 0; i < data->nmbInterventions; i++)
	  for (int t = 0; t <= data->interventions_list[i]->tmax; t++)
	  {
		  key[i][t] = K++;
		  nmbvar++;
	  }
	std::cout << "number of variables : " << nmbvar << std::endl;
    try
    {
		GRBModel model = GRBModel(env);

		// variables
		GRBVar *X = 0;
		X = model.addVars(nmbvar, GRB_BINARY);

        // each intervention is planned
        for (int i = 0; i < data->nmbInterventions; i++)
    	{
        	GRBLinExpr expr = 0;
			for (int j = 0; j < data->T; j++) if(key[i][j] >= 0) expr += X[key[i][j]];
			model.addConstr(expr == 1, "i_" + itos(i));
    	}

        // resources consumption
		for(int r = 0; r < data->nmbResources; r++)
		{
			for(int t = 0; t < data->T; t++)
			{
				GRBLinExpr expr = 0;
				for(int i = 0; i < data->nmbInterventions; i++)
				{
					Intervention* I = data->interventions_list[i];
					if (std::find(I->required_resources.begin(), I->required_resources.end(), r) == I->required_resources.end()) continue;
					for(int j = 0; j < data->T; j++)
					{
						if(key[i][j] < 0) continue;
						int S = j;
						int E = S + I->duration_if_start_at[S];
						if(t >= S && t < E && I->demandRTD[r][t].size() > t - S) expr += (X[key[i][j]] * I->demandRTD[r][t][t - S]);
					}
				}
				model.addConstr(expr >= data->resources_list[r]->min_consumption_at_ts[t] - 9e-6, "min_" + itos(r) + itos(t));
				model.addConstr(expr <= data->resources_list[r]->max_consumption_at_ts[t] + 9e-6, "max_" + itos(r) + itos(t));
			}
		}

		// exclusion
		for(int e = 0; e < data->exclusions.size(); e++)
		{
			int i1 = data->exclusions[e][0];
			int i2 = data->exclusions[e][1];
			int season = data->exclusions[e][2];
			for(int &t : data->season_ts[season])
			{
				GRBLinExpr expr = 0;
				for(int t0 = 0; t0 <= data->interventions_list[i1]->tmax; t0++)
				{
					if(key[i1][t0] < 0) continue;
					if(t0 <= t && t0 + data->interventions_list[i1]->duration_if_start_at[t0] > t) expr += X[key[i1][t0]];
				}
				for(int t0 = 0; t0 <= data->interventions_list[i2]->tmax; t0++)
				{
					if(key[i2][t0] < 0) continue;
					if(t0 <= t && t0 + data->interventions_list[i2]->duration_if_start_at[t0] > t) expr += X[key[i2][t0]];
				}
				model.addConstr(expr <= 1);
			}
		}

    	// OBJECTIVE
        GRBLinExpr obj = 0;
        for (int i = 0; i < data->nmbInterventions; i++)
    	{
			for (int j = 0; j < data->T; j++)
				if(key[i][j] >= 0) obj += (X[key[i][j]] * data->Cost_i_j[i][j]);
    	}
        obj = (2 * data->alpha - 1) * obj;

        if(data->max_nmb_scenarios_at_ts > 1)
        {
			// obj2
        	GRBLinExpr obj2 = 0;
			for(int t = 0; t < data->T; t++)
			{
				GRBVar q = model.addVar(0.0 , GRB_INFINITY , 0.0 , GRB_CONTINUOUS , "q" + itos(t));
				for(int s = 0; s < data->nmb_scenarios_for_ts[t]; s++)
				{
					if(isScenarioRemovedAtTS[t][s]) continue;
					GRBLinExpr exprRiskTS = 0;
					for(int i = 0; i < data->nmbInterventions; i++)
					{
						  Intervention* I = data->interventions_list[i];
						  if(I->is_constant) continue;
						  if(I->is_constant_at_ts[t]) continue;
						  for(int j = 0; j < data->T; j++)
						  {
								if(key[i][j] >= 0 && t >= j && t < j + I->duration_if_start_at[j])
								 if(I->riskTDS[t].size() > t - j && I->riskTDS[t][t - j].size() > s)
									exprRiskTS += (X[key[i][j]] * I->riskTDS[t][t - j][s]);
						  }
					}
					model.addConstr(q >= exprRiskTS, "q" + itos(t) + itos(s));
				}
				obj2 += q / data->T;
			}
			obj += data->beta * obj2;
        }

		model.setObjective(obj, GRB_MINIMIZE);

		if(startFromInitial)
		{
			for(unsigned int i = 0; i < sol->start.size(); i++)
			{
				int t = sol->start[i];
				X[key[i][t]].set(GRB_DoubleAttr_Start, 1.0);
			}
		}

        // SOLVING the model
		model.set(GRB_IntParam_OutputFlag, 0);
		model.set(GRB_DoubleParam_MIPGap, 0.00);
		model.set(GRB_DoubleParam_TimeLimit, MIP_TIME_LIMIT);
		model.set(GRB_IntParam_MIPFocus,  1);
		model.set(GRB_IntParam_Presolve,  1);  // 1 - conservative, 0 - off, 2 - aggressive
		model.set(GRB_IntParam_PrePasses, 1);  // to not spend too much time in pre-solve
        model.set(GRB_IntParam_Method,  1);
		model.optimize();

        double objValue = model.get(GRB_DoubleAttr_ObjVal);
        cout << "   objective : " << objValue << endl;

    	for (int i = 0; i < data->nmbInterventions; i++)
    	{
    		for (int j = 0; j < data->T; j++)
    		{
    	    	if(key[i][j] >= 0 &&  X[key[i][j]].get(GRB_DoubleAttr_X) > 0.99)
    	    	{
    	    		sol->start[i] = j;
    	    	}
    	    }
    	}
   	    delete[] X;
    	return objValue;
     }

	  catch (GRBException e) {
		cout << " Error code = " << e.getErrorCode() << endl << e.getMessage() << endl ;
		return 1e15;
		}
	  catch(...) {
		cout << " Exception during optimization " << endl ;
		return 1e15;
		}

     return 1e15;
}

// NO_OBJECTIVE = true will return first found feasible solution
pair<double, double> MIP_mean(Data* data, Solution* sol, int MIP_TIME_LIMIT, double gap, bool NO_OBJECTIVE)
{
	int nmbvar = 0;
	std::vector<std::vector<int> > key(data->nmbInterventions, std::vector<int>(data->T, -1));
	int K = 0;
	for (int i = 0; i < data->nmbInterventions; i++)
	  for (int t = 0; t <= data->interventions_list[i]->tmax; t++)
	  {
		 key[i][t] = K++;
		 nmbvar++;
	  }

    try
    {
    	GRBModel model = GRBModel(env);
    	GRBVar *X = 0;
    	X = model.addVars(nmbvar, GRB_BINARY);

        // each intervention is planned
        for (int i = 0; i < data->nmbInterventions; i++)
    	{
        	GRBLinExpr expr = 0;
			for (int j = 0; j < data->T; j++) if(key[i][j] >= 0) expr += X[key[i][j]];
			model.addConstr(expr == 1, "i_" + itos(i));
    	}

    	// resources consumption
        for(int r = 0; r < data->nmbResources; r++)
        {
        	for(int t = 0; t < data->T; t++)
        	{
        		GRBLinExpr expr = 0;
            	for(int i = 0; i < data->nmbInterventions; i++)
            	{
            		Intervention* I = data->interventions_list[i];
            		if (std::find(I->required_resources.begin(), I->required_resources.end(), r) == I->required_resources.end()) continue;
            		for(int j = 0; j < data->T; j++)
            		{
            			if(key[i][j] < 0) continue;
						int S = j;
						int E = S + I->duration_if_start_at[S];
						if(t >= S && t < E && I->demandRTD[r][t].size() > t - S) expr += (X[key[i][j]] * I->demandRTD[r][t][t - S]);
            		}
            	}
				model.addConstr(expr >= data->resources_list[r]->min_consumption_at_ts[t] - 9e-6, "min_" + itos(r) + itos(t));
				model.addConstr(expr <= data->resources_list[r]->max_consumption_at_ts[t] + 9e-6, "max_" + itos(r) + itos(t));
        	}
        }

		// exclusion
		for(int e = 0; e < data->exclusions.size(); e++)
		{
			int i1 = data->exclusions[e][0];
			int i2 = data->exclusions[e][1];
			int season = data->exclusions[e][2];
			for(int &t : data->season_ts[season])
			{
				GRBLinExpr expr = 0;
				for(int t0 = 0; t0 <= data->interventions_list[i1]->tmax; t0++)
				{
					if(key[i1][t0] < 0) continue;
					if(t0 <= t && t0 + data->interventions_list[i1]->duration_if_start_at[t0] > t) expr += X[key[i1][t0]];
				}
				for(int t0 = 0; t0 <= data->interventions_list[i2]->tmax; t0++)
				{
					if(key[i2][t0] < 0) continue;
					if(t0 <= t && t0 + data->interventions_list[i2]->duration_if_start_at[t0] > t) expr += X[key[i2][t0]];
				}
				model.addConstr(expr <= 1);
			}
		}

    	// OBJECTIVE
        GRBLinExpr obj = 0;
        if(NO_OBJECTIVE == false)
        {
			for(int i = 0; i < data->nmbInterventions; i++)
				for(int j = 0; j < data->T; j++)
					if(key[i][j] >= 0) obj += (X[key[i][j]] * coefobj[key[i][j]] );
        }
        obj = data->alpha * obj;
        model.setObjective(obj, GRB_MINIMIZE);

		if(NO_OBJECTIVE == false)
		{
			for (int i = 0; i < data->nmbInterventions; i++)
				for (int j = 0; j < data->T; j++)
					if(key[i][j] >= 0)
					{
						if(solinitvector[i] == j) X[key[i][j]].set(GRB_DoubleAttr_Start, 1);
						else X[key[i][j]].set(GRB_DoubleAttr_Start, 0);
					}
		}

		model.set(GRB_DoubleParam_MIPGap, gap);
		model.set(GRB_DoubleParam_TimeLimit, MIP_TIME_LIMIT);
		model.set(GRB_IntParam_OutputFlag, 0);
		model.set(GRB_IntParam_Presolve, 1);  // 1 - conservative, 0 - off, 2 - aggressive
		model.set(GRB_IntParam_PrePasses, 1); // do not spend too much time in pre-solve
		model.optimize();

        double objValue = model.get(GRB_DoubleAttr_ObjVal);
    	for (int i = 0; i < data->nmbInterventions; i++)
    	{
    		for (int j = 0; j < data->T; j++)
    		{
    	    	if(key[i][j] >= 0 &&  X[key[i][j]].get(GRB_DoubleAttr_X) > 0.99)
    	    	{
    	    		sol->start[i] = j;
    	    	}
    	    }
    	}

   	    delete[] X;
    	return pair<double, double>(objValue, model.get(GRB_DoubleAttr_ObjBound));
     }
	  catch (GRBException e) {
		cout << " Error code = " << e.getErrorCode() << endl << e.getMessage() << endl ;
		return pair<double, double>(1e15, 1e15);
		}
	  catch(...) {
		cout << " Exception during optimization " << endl ;
		return pair<double, double>(1e15, 1e15);
		}

	  return pair<double, double>(1e15, 1e15);
}

Data* read_data(string filename)
{
    Data* data = new Data;
    data->datafilename = filename;
    FILE* fp = fopen(filename.c_str(), "r");
    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    Document document;
    document.ParseStream(is);
    fclose(fp);

	data->T = document["T"].GetInt();
	data->Q = document["Quantile"].GetDouble();
	data->alpha = document["Alpha"].GetDouble();
	data->beta = 1 - data->alpha;

    const Value& a = document["Scenarios_number"];
    for(SizeType i = 0; i < a.Size(); i++) {
    	data->nmb_scenarios_for_ts.push_back(a[i].GetInt());
    }

	data->nmbResources = document["Resources"].GetObject().MemberCount();
	int rID = 0;
    for(auto itRes = document["Resources"].GetObject().MemberBegin(); itRes != document["Resources"].GetObject().MemberEnd(); itRes++)
    {
		Resource* R = new Resource;
		R->id = rID++;
		R->name = itRes->name.GetString();
    	auto it = itRes->value.GetObject().MemberBegin();
    	int nmbMins = it->value.Size();
    	auto minlist = it->value.GetArray();
    	for(int i = 0; i < nmbMins; i++) R->min_consumption_at_ts.push_back(minlist[i].GetDouble());
    	it++;
    	int nmbMaxs = it->value.Size();
		auto maxlist = it->value.GetArray();
		for(int i = 0; i < nmbMaxs; i++) R->max_consumption_at_ts.push_back(maxlist[i].GetDouble());
		data->resources_list.push_back(R);
    }

	data->season_ts.resize(4, vector<int>(0));
	int nmbSeasons = document["Seasons"].GetObject().MemberCount();
    for(auto itSeason = document["Seasons"].GetObject().MemberBegin(); itSeason != document["Seasons"].GetObject().MemberEnd(); itSeason++)
    {
    	string name = itSeason->name.GetString();
    	auto list = itSeason->value.GetArray();
    	int index = 0;
    	if(name == "winter") index = 1;
    	if(name == "summer") index = 2;
    	if(name == "is") index = 3;
    	for(int i = 0; i < list.Size(); i++) data->season_ts[index].push_back(atoi(list[i].GetString()) - 1);
    }


    data->nmbInterventions = document["Interventions"].GetObject().MemberCount();

    // do not read all the scenarios if bench is too big (16GB RAM might not be enough)
    // solutions will be feasible, but objective function will not be the same as for the full problem
    bool tooBig = false;
	int maxNmbSc = 0;
    for (int &v: data->nmb_scenarios_for_ts) { if(v > maxNmbSc) maxNmbSc = v; }
    cout << "size " << data->nmbInterventions * data->T * maxNmbSc << endl;
    if(data->nmbInterventions * data->T * maxNmbSc > 22000000) tooBig = true;
    if(tooBig)
    {
        cout << "Warning: bench is too big for 16GB machine - some risks have been ignored!" << endl;
    	for(int t = 0; t < data->T; t++)
    	{
    		data->nmb_scenarios_for_ts[t] = data->nmb_scenarios_for_ts[t] / 2;
    	}
    }


	int iID = 0;
    for(auto itInter = document["Interventions"].GetObject().MemberBegin(); itInter != document["Interventions"].GetObject().MemberEnd(); itInter++)
    {
		Intervention* I = new Intervention;
		I->id = iID++;
		I->name = itInter->name.GetString();

		int longestDur = 0;
		for(auto itc = itInter->value.GetObject().MemberBegin(); itc != itInter->value.GetObject().MemberEnd(); itc++)
		{
			string childname = itc->name.GetString();
			if(childname == "Delta")
			{
				auto list = itc->value.GetArray();
				for(int i = 0; i < list.Size(); i++) {
					int dur = (int) list[i].GetDouble();
					if(dur > longestDur) longestDur = dur;
				}
			}
		}

		I->demandRTD.resize(data->nmbResources, vector<vector<double> >(data->T, vector<double>(longestDur, 0)));
		I->riskTDS.resize(data->T, vector<vector<double> >(longestDur, vector<double>(0)));

		for(auto itc = itInter->value.GetObject().MemberBegin(); itc != itInter->value.GetObject().MemberEnd(); itc++)
		{
			string childname = itc->name.GetString();
			if(childname ==  "tmax")
			{
				I->tmax = atoi(itc->value.GetString())  - 1;
			}
			if(childname == "Delta")
			{
				auto list = itc->value.GetArray();
		    	for(int i = 0; i < list.Size(); i++)  I->duration_if_start_at.push_back((int) list[i].GetDouble());
			}
			if(childname == "workload")
			{
				for(auto itr = itc->value.GetObject().MemberBegin(); itr != itc->value.GetObject().MemberEnd(); itr++)
				{
					string resname = itr->name.GetString();
					int rID = -1;
					for(int r = 0; r < data->resources_list.size(); r++)
					{
						if(data->resources_list[r]->name == resname) { rID = data->resources_list[r]->id; break; }
					}
					if(rID < 0) {cout << "bug1\n"; }
					I->required_resources.push_back(rID);

					for(auto itT1 = itr->value.GetObject().MemberBegin(); itT1 != itr->value.GetObject().MemberEnd(); itT1++)
					{
						int T1 = atoi(itT1->name.GetString());
						for(auto itT2 = itT1->value.GetObject().MemberBegin(); itT2 != itT1->value.GetObject().MemberEnd(); itT2++)
						{
							int T2 = atoi(itT2->name.GetString());
							double dem = itT2->value.GetDouble();
							int D = T1 - T2;
							while(I->demandRTD[rID][T1 - 1].size() <= D) I->demandRTD[rID][T1 - 1].push_back(0);
							I->demandRTD[rID][T1 - 1][D] = dem;
						}
					}
				}
			}
			if(childname == "risk")
			{
				for(auto itT1 = itc->value.GetObject().MemberBegin(); itT1 != itc->value.GetObject().MemberEnd(); itT1++)
				{
					int T1 = atoi(itT1->name.GetString());
					for(auto itT2 = itT1->value.GetObject().MemberBegin(); itT2 != itT1->value.GetObject().MemberEnd(); itT2++)
					{
						int T2 = atoi(itT2->name.GetString());
						auto list = itT2->value.GetArray();
						for(int i = 0; i < list.Size(); i++)
						{
							if(I->riskTDS[T1 - 1][T1 - T2].size() < data->nmb_scenarios_for_ts[T1 - 1])
							   I->riskTDS[T1 - 1][T1 - T2].push_back(list[i].GetDouble());
						}
					}
				}
			}
		}

		I->has_same_risk_over_all_scenarios.resize(data->T, vector<bool>(data->T, true));
		for(int t1 = 0; t1 < data->T; t1++)
		  for(int t2 = 0; t2 <= t1; t2++)
		  {
			  if(I->riskTDS[t1].size() <= t1 - t2) continue;
			  if(I->riskTDS[t1][t1 - t2].size() < 2) continue;
			  for(int s = 1; s < data->nmb_scenarios_for_ts[t1]; s++)
				  if(I->riskTDS[t1][t1 - t2].size() > s && equal(I->riskTDS[t1][t1 - t2][0], I->riskTDS[t1][t1 - t2][s]) == false)
				  {
					  I->has_same_risk_over_all_scenarios[t1][t2] = false;
					  break;
				  }
		  }

		I->is_constant = true;
		I->is_constant_at_ts.resize(data->T, true);
		for(int t1 = 0; t1 < data->T; t1++)
		{
			bool isconstantatts = true;
			for(int t2 = 0; t2 < data->T; t2++)
			{
				if(I->has_same_risk_over_all_scenarios[t1][t2] == false)
				{
					I->is_constant = false;
					isconstantatts = false;
					break;
				}
			}
			I->is_constant_at_ts[t1] = isconstantatts;
		}

		data->interventions_list.push_back(I);
		I->belongsToAnyExclusion = false;
	}


    int Exc = document["Exclusions"].GetObject().MemberCount();
	for(auto itE = document["Exclusions"].GetObject().MemberBegin(); itE != document["Exclusions"].GetObject().MemberEnd(); itE++)
	{
		string exname = itE->name.GetString();
		auto list = itE->value.GetArray();
		vector<int> exc;
		for(int i = 0; i < list.Size(); i++)
		{
			string s = list[i].GetString();
			if(s == "full"  ) exc.push_back(FULL_SEASON);
			if(s == "winter") exc.push_back(WINTER_SEASON);
			if(s == "summer") exc.push_back(SUMMER_SEASON);
			if(s == "is"    ) exc.push_back(INTER__SEASON);
			for(int r = 0; r < data->interventions_list.size(); r++)
				if(data->interventions_list[r]->name == list[i].GetString())
				{
					exc.push_back(data->interventions_list[r]->id);
					data->interventions_list[r]->belongsToAnyExclusion = true;
					break;
				}
		}
		data->exclusions.push_back(exc);
	}

	for (int &v: data->nmb_scenarios_for_ts)
	{
		if(v < data->min_nmb_scenarios_at_ts) data->min_nmb_scenarios_at_ts = v;
		if(v > data->max_nmb_scenarios_at_ts) data->max_nmb_scenarios_at_ts = v;
	}

	data->precalculate_stuff();

	//document.Clear();	//document.SetObject();	//document.Swap(Value(kObjectType).Move());

	return data;
}

// randomness
#define MBIG    1000000000L
#define MSEED   161803398L
#define MZ      0
static long ma[NUM_THREADS][56];
static int inext[NUM_THREADS], inextp[NUM_THREADS];
void srand_multithread(int g, int threadID)
{
	long mj, mk;
	int i, ii, k;
	mj=MSEED-g;
	mj %= MBIG;
	ma[threadID][55]=mj;
	mk=1;
	for (i=1; i<=54 ; i++)
		{
		ii = (21 * i) % 55;
		ma[threadID][ii]=mk;
		mk=mj-mk;
		if (mk < MZ)
			mk += MBIG;
		mj=ma[threadID][ii];
		}
	for (k=1 ; k<=4 ; k++)
		for (i=1 ; i<=55 ; i++)
			{
			ma[threadID][i] -= ma[threadID][1 + (i+30) % 55];
			if (ma[threadID][i] < MZ)
			ma[threadID][i] += MBIG;
			}
	inext[threadID]=0;
	inextp[threadID]=31;
}
long rand_multithread(int threadID)
{
	long mj;
	if (++inext[threadID] == 56)
		inext[threadID]=1;
	if (++inextp[threadID] == 56)
		inextp[threadID]=1;
	mj=ma[threadID][inext[threadID]] - ma[threadID][inextp[threadID]];
	if (mj < MZ)
		mj += MBIG;
	ma[threadID][inext[threadID]]=mj;
	if (mj < 0) return -mj;
	return mj;
}
