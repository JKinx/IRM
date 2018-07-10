import numpy as np
from collections import Counter, defaultdict
from scipy.special import beta as Beta
import numpy.random as npr
from copy import deepcopy
from tqdm import tqdm
from sortedcontainers import SortedSet
from enum import Enum

class Sampler(Enum):
    GIBBS = 0
    CLIMB = 1
    ANNEAL = 2

class model(object):
    """Infinite Relational Model."""
    def __init__(self, gamma, alpha, beta, relations):
        # CRP gamma param for z assignments
        self.gamma = gamma
        
        # beta param for η
        self.beta = beta
        self.alpha = alpha
        
        # list of relations 
        # a relation is a tuple of type (object,object,bool)
        self.relations = relations
        
        # list of unique objects
        objs = set()
        for (a,b,_) in self.relations:
            objs.add(a)
            objs.add(b)
        self.objs  = list(objs)
        
        # total number of objects
        self.num_objs = len(self.objs)
        
        # denominator of CRP prob for joint table assignment (γ)(γ+1)...(γ+n-1)
        # self.CRProb_joint_denom = np.prod(np.arange(self.gamma, self.gamma + self.num_objs, 1))
        self.CRProb_joint_denom = np.prod(np.arange(self.gamma, self.gamma + self.num_objs))
        
        # denominator of CRP prob for last customer's table assignment (γ+n-1)
        self.CRProb_cond_denom = self.gamma + self.num_objs - 1
        
        self.cluster_ids = SortedSet(range(self.num_objs))
        # cluster assignements for objects, initialized to 0 for all
        # self.z = np.zeros(self.num_objs, dtype="int")
        self.z = {}
        
        cluster_id = self.cluster_ids.pop(index = 0)

        # set of all object types 
        self.obj_types = set([cluster_id])

        self.obj_type_mem = defaultdict(set)
        
        for obj in self.objs:
            self.z[obj] = cluster_id
            self.obj_type_mem[cluster_id].add(obj)
        
        # number of objects in each object type
        self.obj_type_count = Counter()
        self.obj_type_count[cluster_id] += self.num_objs
        
        # graph of relations described by self.relations
        # each object has four lists: 0 - outgoing with 0, 1 - outgoing with 1,
        # 2 - incoming with 0, 3 - incoming in 1, 4 - does the object interact with itself?
        self.graph = {}
        
        for obj in self.objs:
            self.graph[obj] = [[] for _ in range(4)]
            self.graph[obj].append(None)
    
        # counts of 0 and 1 edges between different pairs of object types
        # 0 - m_bar, 1 - m
        self.m = Counter()
        self.m_bar = Counter()
        
        # keep track of (object, object) pairs
        # allows for faster calculation of P(R|z)
        self.pair_count = Counter()
        self.pairs = set([(0,0)])
        
        # build graph and update counters
        for (i, j, b) in self.relations:
            if i == j:
                self.graph[i][4] = b
            elif b:
                self.graph[i][1].append(j)
                self.graph[j][3].append(i)
            else:
                self.graph[i][0].append(j)
                self.graph[j][2].append(i)
                
            if b:
                self.m[(0,0)] += 1
            else:
                self.m_bar[(0,0)] += 1
            
            self.pair_count[(0,0)] += 1
        
        # maximum joint prob as measure of max posterior 
        # p(z|R) ∝ p(z, R) = p(R|z)p(z)
        self.max_prob = -np.inf
        self.best_z = deepcopy(self.z)
    
    def backup(self):
        """Create a backup of current state."""
        self.backup_cluster_ids = deepcopy(self.cluster_ids)
        self.backup_z = deepcopy(self.z)
        self.backup_obj_types = deepcopy(self.obj_types)
        self.backup_obj_type_count = deepcopy(self.obj_type_count)
        self.backup_obj_type_mem = deepcopy(self.obj_type_mem)
        self.backup_m = deepcopy(self.m)
        self.backup_m_bar = deepcopy(self.m_bar)
        self.backup_pair_count = deepcopy(self.pair_count)
        self.backup_pairs = deepcopy(self.pairs)
        
    def restore(self):
        """Restore state to backup configurations."""
        self.cluster_ids = deepcopy(self.backup_cluster_ids)
        self.z = deepcopy(self.backup_z)
        self.obj_types = deepcopy(self.backup_obj_types)
        self.obj_type_count = deepcopy(self.backup_obj_type_count)
        self.obj_type_mem = deepcopy(self.backup_obj_type_mem)
        self.m = deepcopy(self.backup_m)
        self.m_bar = deepcopy(self.backup_m_bar)
        self.pair_count = deepcopy(self.backup_pair_count)
        self.pairs = deepcopy(self.backup_pairs)
        
    def empty_backup(self):
        """Empty backup."""
        del self.backup_cluster_ids, self.backup_z, self.backup_obj_types
        del self.backup_obj_type_count, self.backup_m, self.backup_m_bar
        del self.backup_pair_count, self.backup_pairs, self.backup_obj_type_mem
        
    def obj_type_inc(self, obj_type):
        """Increment the number of objects with given type.
        
        Add to set of object types if new.
        """
        self.obj_type_count[obj_type] += 1
        if self.obj_type_count[obj_type] == 1:
            self.obj_types.add(obj_type)
            self.cluster_ids.remove(obj_type)
    
    def obj_type_dec(self, obj_type):
        """Decrement the number of objects with given type.
        
        Remove from set of object types if no objects belong to type.
        """
        self.obj_type_count[obj_type] -= 1
        if self.obj_type_count[obj_type] == 0:
            self.cluster_ids.add(obj_type)
            self.obj_types.remove(obj_type)
            del self.obj_type_count[obj_type]
            del self.obj_type_mem[obj_type]
            
    def pair_count_update(self, pair, count):
        """Update the number of object type pairs with edges.
        
        This allows us to more efficiently compute p(R|z) which needs 
        iteration through all object type pairs. By only iterating 
        through pairs with non-zero m or m_bar values, we potentially
        save time.
        """
        self.pair_count[pair] += count
        
        if self.pair_count[pair] > 0 and self.pair_count[pair] == count:
            self.pairs.add(pair)
        elif self.pair_count[pair] == 0:
            self.pairs.remove(pair)
            del self.pair_count[pair]
    
    def update_ms(self, m, pair, count):
        """Update the number of object type pairs with edges of type 1
        (m) and type 0(m_bar).
        """
        if m:
            self.m[pair] += count
            if self.m[pair] == 0:
                del self.m[pair]
        else:
            self.m_bar[pair] += count
            if self.m_bar[pair] == 0:
                del self.m_bar[pair]
    
        self.pair_count_update(pair, count)
            
    def deassign(self, obj):
        """Deassign given object.
        
        This is done in preparation for sampling the cluster assignment
        of the given object. All graph counters are updated correspondingly.
        """
        curr = obj
        curr_type = self.z[obj]

        m = Counter()
        m_bar = Counter()
        
        self.z[curr] = None
        self.obj_type_mem[curr_type].remove(obj)
        self.obj_type_dec(curr_type)

        if self.graph[curr][4] == 1:    
            self.update_ms(1, (curr_type, curr_type), -1)
            m[(curr_type, curr_type)] -= 1
        elif self.graph[curr][4] == 0:
            self.update_ms(0, (curr_type, curr_type), -1)
            m_bar[(curr_type, curr_type)] -= 1
        
        for obj in self.graph[curr][0]:
            typ = self.z[obj]
            self.update_ms(0, (curr_type, typ), -1)
            m_bar[(curr_type, typ)] -= 1
        for obj in self.graph[curr][1]:
            typ = self.z[obj] 
            m[(curr_type, typ)] -= 1
            self.update_ms(1, (curr_type, typ), -1)
        for obj in self.graph[curr][2]:
            typ = self.z[obj]
            self.update_ms(0, (typ, curr_type), -1)
            m_bar[(typ, curr_type)] -= 1
        for obj in self.graph[curr][3]:
            typ = self.z[obj] 
            self.update_ms(1, (typ, curr_type), -1)
            m[(typ, curr_type)] -= 1

        return {"obj" : curr, "old_type" : curr_type, "new_type" : None, "m" : m, "m_bar" : m_bar}
         
    def assign(self, obj, new_type):
        """Assign given object 'obj' to type 'new_type'.
        
        This function returns a dictionary called hist which is a log
        of changes made when the assignment happens. It allows undoing
        or redoing the particular assignment.
        """
        curr = obj

        m = Counter()
        m_bar = Counter()
    
        self.z[curr] = new_type
        self.obj_type_mem[new_type].add(curr)
        self.obj_type_inc(new_type) 
        
        if self.graph[curr][4] == 1:
            self.update_ms(1, (new_type, new_type), 1)
            m[(new_type, new_type)] += 1     
        elif self.graph[curr][4] == 0:
            self.update_ms(0, (new_type, new_type), 1)
            m_bar[(new_type, new_type)] += 1
            
        for obj in self.graph[curr][0]:
            typ = self.z[obj]
            self.update_ms(0, (new_type, typ), 1)
            m_bar[(new_type, typ)] += 1
        for obj in self.graph[curr][1]:
            typ = self.z[obj]
            self.update_ms(1, (new_type, typ), 1)
            m[(new_type, typ)] += 1
        for obj in self.graph[curr][2]:
            typ = self.z[obj]
            self.update_ms(0, (typ, new_type), 1)
            m_bar[(typ, new_type)] += 1
        for obj in self.graph[curr][3]:
            typ = self.z[obj]
            self.update_ms(1, (typ, new_type), 1)
            m[(typ, new_type)] += 1
        
        return {"obj" : curr, "old_type" : None, "new_type" : new_type, "m" : m, "m_bar" : m_bar}
    
    def revert(self, hist):
        """Undo the assignment logged by hist."""
        self.z[hist["obj"]] = hist["old_type"]

        if hist["new_type"] != None:
            self.obj_type_mem[hist["new_type"]].remove(hist["obj"])
            self.obj_type_dec(hist["new_type"])
        if hist["old_type"] != None:
            self.obj_type_mem[hist["old_type"]].add(hist["obj"])
            self.obj_type_inc(hist["old_type"])
        
        for pair in hist["m"]:
            self.update_ms(1, pair, -hist["m"][pair])   
        for pair in hist["m_bar"]:
            self.update_ms(0, pair, -hist["m_bar"][pair])
    
    def reassign(self, hist):
        """Redo the assignment logged by hist."""
        self.z[hist["obj"]] = hist["new_type"]
        if hist["new_type"] != None:
            self.obj_type_mem[hist["new_type"]].add(hist["obj"])
            self.obj_type_inc(hist["new_type"])
        if hist["old_type"] != None:
            self.obj_type_mem[hist["old_type"]].remove(hist["obj"])
            self.obj_type_dec(hist["old_type"])
        
        for pair in hist["m"]:
            self.update_ms(1, pair, hist["m"][pair])
        for pair in hist["m_bar"]:
            self.update_ms(0, pair, hist["m_bar"][pair])
    
    def Beta_factor(self, a, b):
        """Beta(m(a,b) + β, mbar(a,b) + β)/Beta(β,β)"""
        beta = self.beta
        alpha = self.alpha
        return Beta(self.m[(a,b)] + alpha, self.m_bar[(a,b)] + beta) / Beta(alpha, beta)
        
    def CRProb(self, obj, obj_type):
        """p(z[obj] = obj_type | z[-obj]).
        
        z[-obj] : all z except z[obj].
        Conditional Probability of object 'obj' being in cluster 'obj_type'
        as given by by the CRP distribution. 
        """
        if obj_type not in self.obj_types:
            return self.gamma / self.CRProb_cond_denom

        prob = self.obj_type_count[obj_type]/ self.CRProb_cond_denom
        return prob
    
    def Prob_Rz(self):
        """p(R|z)"""
        prob = 1
        for (a, b) in self.pairs:
            prob *= self.Beta_factor(a, b)
        return prob
    
    def Prob_zi(self, obj, obj_type):
        """p(z[obj] = obj_type | R, z[-obj]).
        
        p(z_{i}=a|z_{-i},R) ∝ p(R|z)p(z_{i}=a|z_{-i})
        """
        prob = self.CRProb(obj, obj_type)
        
        # assign obj to obj_type, compute probability and undo assignment
        hist = self.assign(obj, obj_type)
        prob *= self.Prob_Rz()
        self.revert(hist)
        return prob, hist
    
    def Prob_z(self):
        """p(z)"""
        prob = 1
        for typ in self.obj_type_count:
            num = self.obj_type_count[typ]
            prob *= self.gamma
            prob *= np.prod(np.arange(1, num))
        prob /= self.CRProb_joint_denom
        return prob
        
    def Prob_joint(self):
        """p(z,R) = p(R|z)p(z).
        
        When calculating π(z*)/π(z) = p(z*|R)/p(z|R) for Metropolis-Hasting update,
        we use p(z*,R)/p(z|R) since p(R) in denoms cancel one another.
        """
        prob = self.Prob_Rz()
        prob *= self.Prob_z()
        return prob
        
    def sample_zi(self, obj, alg = Sampler.GIBBS, restricted = False, type_opts = None, temp = None):
        """Sample cluster assignment for object 'obj'.
        
        Possible algorithms : Gibbs, Hill-climb, Simulated annealing
        Clutser assignment can be restricted to those in 'type_opts'.
        """
        types = []
        probs = []
        hists = {}
        
        # deassign obj to prepare for sampling
        if alg == Sampler.ANNEAL:
            old_prob = self.Prob_joint()
            log = self.deassign(obj)
        else:
            self.deassign(obj)
        
        if not restricted:
            # compute probabilities p(z_{obj}|z_{-obj},R) for existing clusters
            for typ in self.obj_types:
                prob, hist = self.Prob_zi(obj, typ)
                types.append(typ)
                probs.append(prob)
                hists[typ] = hist

            # compute probability p(z_{obj}|z_{-obj},R) for new clusters
            new_table = self.cluster_ids[0]
            prob, hist = self.Prob_zi(obj, new_table)
            types.append(new_table)
            probs.append(prob)
            hists[new_table] = hist
        else:
            for typ in type_opts:
                prob, hist = self.Prob_zi(obj, typ)
                types.append(typ)
                probs.append(prob)
                hists[typ] = hist
        
        # normalize probabilities to make sum 1
        probs /= sum(probs)
        
        # sample object type and reassign obj to sampled type
        if alg == Sampler.CLIMB:
            sample_id = np.argmax(probs)
        else:
            sample_id = npr.choice(np.arange(len(probs)), size = 1, p = probs)[0]
        sample = types[sample_id]

        self.reassign(hists[sample])

        if alg == Sampler.ANNEAL:
            new_prob = self.Prob_joint()
            ap = np.exp((new_prob - old_prob)/temp)
            if ap <= npr.random():
                self.revert(hists[sample])
                self.revert(log)

        # compute joint and update max prob if necessary
        joint = self.Prob_joint()
        if joint > self.max_prob:
            self.max_prob = joint
            self.best_z = deepcopy(self.z)
        
        return probs[sample_id]                       
    
    def sim_sample_zi(self, obj, type_ori, type_opts):
        """Sample cluster assignment for object 'obj'."""
        types = []
        probs = []
        hists = {}
        
        # deassign obj to prepare for sampling
        self.deassign(obj)
        
        # get probabilities for assigning obj
        for typ in type_opts:
            prob, hist = self.Prob_zi(obj, typ)
            types.append(typ)
            probs.append(prob)
            hists[typ] = hist
                
        # normalize probabilities to make sum 1
        probs /= sum(probs)
        
        # reassign obj to original type 'type_ori'
        self.reassign(hists[type_ori])
        
        return probs[type_opts.index(type_ori)] 

    def climb_scan(self):
        """Greedy type assignment for objects.

        Iteratively assign each object to best possible cluster given
        current state.
        """
        for obj in self.objs:
            self.sample_zi(obj, alg = Sampler.CLIMB)

    def gibbs_scan(self, restricted, objs, type_opts):
        """A single gibbs sampling scan.
        
        If restricted, implement a restricted sampling scan assigning
        objs to one of type_opts.
        """
        prob = 1
        if restricted:
            for obj in objs:
                prob *= self.sample_zi(obj, restricted = restricted, type_opts = type_opts)
        else:
            for obj in self.objs:
                prob *= self.sample_zi(obj, restricted = restricted, type_opts = type_opts)
        return prob

    def anneal_scan(self, temp):
        """A single annealed sampling scan."""
        for obj in self.objs:
            self.sample_zi(obj, alg = Sampler.ANNEAL, temp = temp)
        
    def get_z(self):
        return self.z
    
    def get_S(self, obj_types, exclude):
        """Get all objs belonging to either of obj_types.
        
        Exclude objects in exclude.
        """
        objs = []
        Sij = deepcopy(self.obj_type_mem[obj_types[0]])
        if obj_types[0] != obj_types[1]:
            Sij |= self.obj_type_mem[obj_types[1]]

        for obj in self.z:
            if obj not in exclude and self.z[obj] in obj_types:
                objs.append(obj) 
        return objs

    def rand_obj(self, z_i, num_objs):
        """Sample num_objs unique objects from type z_i."""
        return npr.choice(list(self.obj_type_mem[z_i]), size = num_objs, replace = False)
    
    def rand_z(self, restricted = False, objs = None, options = None):
        """Randomly assign objs to one of options.
        
        If not restricted, assign to each object a random type.
        """
        if not restricted:
            for obj in self.objs:
                typ = npr.randint(self.num_objs)
                self.deassign(obj)
                self.assign(obj, typ)
        else:
            for obj in objs:
                typ = npr.choice(options)
                self.deassign(obj)
                self.assign(obj, typ)
    
    def merge_sm(self, obj_ij, obj_types, t):
        """Implement merge process for Metropolis-Hastings Update."""
        # remember orginal assignments z
        # required to compute q(z|z_merge)
        ori_z = deepcopy(self.z)
        
        # perform t restricted gibbs sampling for objects in S
        S = self.get_S(obj_types, obj_ij)
        self.rand_z(True, S, obj_types)
        self.sample_restr(t, S, obj_types)
        
        # compute q(z|z_merge)
        prob = 1
        for obj in S:
            prob *= self.sim_sample_zi(obj, ori_z[obj], obj_types)
        
        # merge all objects in S∪{obj_i, obj_j} by assigning all to z_j
        self.deassign(obj_ij[0])
        self.assign(obj_ij[0], obj_types[1])
        for obj in S:
            if self.z[obj] != obj_types[1]:
                self.deassign(obj)
                self.assign(obj, obj_types[1])
            
        # return ratio of transitional probabilities 
        # q(z|z_merge)/q(z_merge/z) = q(z|z_merge)
        return prob


    def merge_opt(self, z_i, z_j):
        """Merge objects with types z_i and z_j into single cluster with type z_i."""
        S = self.get_S((z_j, z_j), [])

        for obj in S:
            self.deassign(obj)
            self.assign(obj, z_i)

    def split_sm(self, obj_ij, z_j, t):
        """Implement merge process for Metropolis-Hastings Update."""
        # assign obj_i to a new cluster
        z_i = self.cluster_ids[0]
        self.deassign(obj_ij[0])
        self.assign(obj_ij[0], z_i)
        
        # perform t restricted gibbs sampling for objects in S
        S = self.get_S((z_i,z_j), obj_ij)
        self.rand_z(True, S, (z_i,z_j))
        self.sample_restr(t, S, (z_i,z_j))
        
        # return ratio of transitional probabilities 
        # q(z|z_split)/q(z_split/z) = 1/q(z|z_split)
        return 1 / self.gibbs_scan(True, S, (z_i,z_j))

    def split_opt(self, obj_ij, z_i):
        """Split the objects in type z_i into two clusters.

        Use obj_i and obj_j as reference objects for the types
        """
        z_j = self.cluster_ids[0]
        self.deassign(obj_ij[1])
        self.assign(obj_ij[1], z_j)

        S = self.get_S((z_i,z_j), obj_ij)
        self.rand_z(True, S, (z_i,z_j))
        
    def split_merge(self, t):
        """split-merge Metroplolis-Hastings Update.
        
        t = number of intermediate restricted Gibbs sampling scans.
        """
        # compute π(z) = p(z|R) ∝ p(z,R)
        pi_z = self.Prob_joint()
        
        # backup current state
        self.backup()
        
        # randomly choose two objects
        obj_i, obj_j = npr.choice(self.objs, size = 2, replace = False)
        z_i, z_j = self.z[obj_i], self.z[obj_j]
        
        # execute split or merge 
        if z_i == z_j:
            trans_prob = self.split_sm((obj_i, obj_j), z_j, t)
        else:
            trans_prob = self.merge_sm((obj_i, obj_j), (z_i,z_j), t)
        
        # compute π(z*) = p(z*|R) ∝ p(z*,R)
        pi_z_new = self.Prob_joint()
        
        # a(z*,z) = min(1, transitional probability * π(z*) / π(z))
        accep_prob = min(1, trans_prob * pi_z_new / pi_z)
        
        # accept or reject
        accept = npr.binomial(1, accep_prob)
        if accept:
            self.empty_backup()
        else:
            self.restore()
    
    def opt_split(self, alg = Sampler.CLIMB, temp = None):
        """Optimization step : Attempt splitting each class.

        In order to run fast enough for big datasets, we randomly choose a 
        a type num_types times weighted by the cluster size.
        Possible techniques : hill-climbing, simulated annealing.
        """
        types = []
        sizes = []
        num_types = len(self.obj_types)

        for typ in self.obj_types:
            types.append(typ)
            sizes.append(self.obj_type_count[typ])

        # normalize cluster sizes to sum to 1
        sizes /= np.sum(sizes)

        for _ in range(num_types):
            old_prob = self.Prob_joint()
            self.backup()

            # choose an object type 
            z_i = npr.choice(types, size = 1, p = sizes)[0]

            if self.obj_type_count[z_i] == 1:
                continue

            # pick two objects from the type and split using them
            obj_i, obj_j = self.rand_obj(z_i, 2)
            self.split_opt((obj_i, obj_j), z_i)

            new_prob = self.Prob_joint()

            if alg == Sampler.CLIMB:
                if new_prob <= old_prob:
                    self.restore()
            else:
                ap = np.exp((new_prob - old_prob)/temp)
                if ap <= npr.random():
                    self.restore()
            self.empty_backup()

    def opt_merge(self, alg = Sampler.CLIMB, temp = None):
        """Optimization step : Attempt merging classes with one other.

        In order to run fast enough for big datasets, for each type we 
        randomly choose another type and attempt merge.
        Possible techniques : hill-climbing, simulated annealing.
        """
        if len(self.obj_types) == 1:
            return

        types = list(deepcopy(self.obj_types))
        for z_i in types:
            z_j = z_i

            while z_j == z_i:
                z_j = npr.choice(types)

            old_prob = self.Prob_joint()
            self.backup()

            self.merge_opt(z_i, z_j)
            new_prob = self.Prob_joint()

            if alg == Sampler.CLIMB:
                if new_prob <= old_prob:
                    self.restore()
            else:
                ap = np.exp((new_prob - old_prob)/temp)
                if ap <= npr.random():
                    self.restore()
            self.empty_backup()

    def sample_gibbs(self, num_iters):
        """Run num_iters gibbs sampling scans."""
        for i in range(num_iters):
            a.gibbs_scan(False, None, None)
    
    def sample_restr(self, num_iters, objs, type_opts):
        """Run num_ites restricted gibbs sampling scans."""
        for i in range(num_iters):
            a.gibbs_scan(True, objs, type_opts)

    def sample_climb(self, num_iter):
        """Sample using hillclimbing with random restarts"""
        old_prob = -np.inf
        new_prob = -np.inf
        num_repeats = 0

        for _ in tqdm(range(num_iter)):
            # move each class to the best type
            self.climb_scan()

            # attempt cluster splits
            self.opt_split()

            # attempt cluster merges
            self.opt_merge()

            new_prob = np.log(self.Prob_joint())

            # if no substantial changes for last 8 iters, restart with
            # random cluster assignments
            if abs(new_prob - old_prob) < 0.00001 * abs(new_prob):
                num_repeats += 1
                if num_repeats == 8:
                    num_repeats = 0
                    self.rand_z()
            else:
                num_repeats = 0

            old_prob = new_prob

    def sample_anneal(self, num_iter):
        """Sample using simulated annealing"""
        T = 1.0
        T_min = 0.0001
        rate = 0.9

        while T > T_min:
            for _ in range(num_iter):
                self.anneal_scan(T)
                self.opt_split(Sampler.ANNEAL, T)
                self.opt_merge(Sampler.ANNEAL, T)
            T *= rate

    def sample_full(self, num_iter, inter, split_merge, gibbs):
        """Sample using both Metropolis-Hastings update and Gibbs 
        sampling Scans.
        
        num_iters = number of complete sampling iterations
        iter = number of intermediate restricted gibbs samplings
               per split-merge update
        split_merge = number of split-merge update per iteration
        gibbs = number of gibbs sampling scans per iteration
        """
        for _ in tqdm(range(num_iter)):
            for _ in range(split_merge):
                self.split_merge(inter)
            self.sample_gibbs(gibbs)