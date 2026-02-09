import yaml
import rtamt
import sys

class RTAMTYmlParser: 
    
    def __init__(self, config_path: str):
        
        with open(config_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        # We will store a dedicated, pre-compiled monitor for each rule.
        # This avoids re-parsing the formula at every training step (huge performance boost).
        self.monitors = {}
        
        self.timestep = 1 if 'timestep' not in config_dict.keys() else config_dict['timestep']
        self.horizon_length = 1 if 'horizon' not in config_dict.keys() else config_dict['horizon']

        # Pull the reward type from the config
        self.dense = False if 'dense' not in config_dict.keys() else config_dict['dense']

        self.stl_variables = config_dict['variables']
        self.specifications = config_dict['specifications']
        
        # Helper function to initialize a fresh monitor with all variables declared.
        # This replaces the old single-monitor setup to allow independent rule evaluation.
        def create_base_monitor():
            spec = rtamt.StlDiscreteTimeOfflineSpecification()
            
            # Sort through specified constants that will be used in the specifications
            if 'constants' in config_dict.keys():
                for c in config_dict['constants']:
                    spec.declare_const(c['name'], c['type'], c['value'])
            
            # Sort through specified variables that will be tracked
            for v in self.stl_variables:
                spec.declare_var(v['name'], v['type'])
                
                # IMPORTANT: Explicitly set IO type for RTAMT 0.3.5 compatibility
                if 'i/o' in v.keys():
                    spec.set_var_io_type(v['name'], v['i/o'])
                elif v.get('location') == 'obs':
                    spec.set_var_io_type(v['name'], 'input')
            return spec

        # Collect specifications and compile monitors
        for i in self.specifications:
            rule_name = i['name']
            raw_spec = i['spec']
            
            # Extract pure formula (remove "RuleName = " prefix if present)
            if '=' in raw_spec:
                formula = raw_spec.split('=', 1)[1].strip()
            else:
                formula = raw_spec
            
            try:
                # Create a dedicated monitor instance for this rule
                monitor = create_base_monitor()
                monitor.spec = formula
                monitor.parse() # Compile ONCE at startup
                
                # Store in dictionary
                self.monitors[rule_name] = monitor
                
            except rtamt.StlParseException as err:
                print(f'STL Parse Exception for rule {rule_name}: {err}')
                sys.exit()

        # Initialize data structure for inputs (Legacy support)
        self.data = dict()
        self.data['time'] = []

    def get_non_global_rules(self):
        non_global = list(filter(lambda spec: spec["name"] != 'global' , self.specifications))
        names = list(map(lambda spec: spec['name'], non_global))
        return names
            
    # This is the Professor's original method. 
    # NOTE: It will not work with the current setup (self.stl_spec is missing) 
    # and is kept here strictly for reference/comparison.
    def old_compute_robustness_dense(self, tau_state):
        
        # assert len(tau_state) == self.horizon_length, "Tau state does not match STL rules horizon"
        data = {}
        for key in self.data.keys():
            data[key] = []
        for obs in tau_state:
            for i in self.stl_variables:
                if i['location'] == 'obs':
                    data[i['name']].append(obs[i['identifier']])
        data['time'] = range(len(tau_state))
        
        single_rho = {}
        total_rho = 0
        if self.dense:
            _ = self.stl_spec.evaluate(data)
            for i in self.specifications:
                val = self.stl_spec.get_value(i['name'])[0]
                single_rho[i['name']] = val 
                total_rho += float(i['weight']) * val
                
        return total_rho, single_rho

    def compute_robustness_dense(self, tau_state):
        # --- OPTIMIZED EVALUATION ---
        # No parsing here. Just data preparation and evaluation.
        
        # 1. Prepare Data (One-time formatting for all monitors)
        data = {}
        steps = len(tau_state)
        
        # Create 'time' list as pure floats
        data['time'] = [float(t) for t in range(steps)]

        # Fill input variables (force float cast)
        for i in self.stl_variables:
            if i['location'] == 'obs':
                data[i['name']] = [float(obs[i['identifier']]) for obs in tau_state]
        
        single_rho = {}
        total_rho = 0.0
        
        if self.dense:
            # Iterate over pre-compiled monitors
            for i in self.specifications:
                name = i['name']
                weight = float(i.get('weight', 1.0))
                
                try:
                    # Retrieve the pre-compiled monitor
                    monitor = self.monitors[name]
                    
                    # Evaluate directly (Fast C++ execution)
                    res = monitor.evaluate(data)
                    
                    # --- EXTRACT VALUE AT t=0 ---
                    val = 0.0
                    if isinstance(res, list) and len(res) > 0:
                        # Case: [[t0, v0], [t1, v1]...] -> take v0
                        val = float(res[0][1])
                    elif isinstance(res, (float, int)):
                        # Case: scalar
                        val = float(res)
                    
                    # Store results
                    single_rho[name] = val 
                    total_rho += weight * val
                    
                except Exception as e:
                    # Fallback only if runtime error occurs
                    # print(f"RTAMT Runtime Error rule '{name}': {e}")
                    single_rho[name] = 0.0
                
        return total_rho, single_rho