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
                
        self.stl_spec = rtamt.StlDiscreteTimeSpecification()
        self.data = dict()
        self.data['time'] = []
        
        self.timestep = 1 if 'timestep' not in config_dict.keys() else config_dict['timestep']
        self.horizon_length = 1 if 'horizon' not in config_dict.keys() else config_dict['horizon']

        # Pull the reward type from the config
        self.dense = False if 'dense' not in config_dict.keys() else config_dict['dense']
        # Sort through specified constants that will be used in the specifications
        if 'constants' in config_dict.keys():
            constants = config_dict['constants']
            for i in constants:
                self.stl_spec.declare_const(i['name'], i['type'], i['value'])

        # Sort through specified variables that will be tracked
        self.stl_variables = config_dict['variables']
        for i in self.stl_variables:
            self.stl_spec.declare_var(i['name'], i['type'])
            self.data[i['name']] = []
            if 'i/o' in i.keys():
                self.stl_spec.set_var_io_type(i['name'], i['i/o'])

        # Collect specifications
        self.specifications = config_dict['specifications']
        spec_str = "out = "
        for i in self.specifications:
            #self.stl_spec.declare_var(i['name'], 'float')
            self.stl_spec.add_sub_spec(i['spec'])
            spec_str += i['name'] + ' and '
            if 'weight' not in i.keys():
                i['weight'] = 1.0
        spec_str = spec_str[:-5]
        self.stl_spec.declare_var('out', 'float')
        self.stl_spec.spec = spec_str

        # Parse the specification
        try:
            self.stl_spec.parse()
        except rtamt.StlParseException as err:
            print('STL Parse Exception: {}'.format(err))
            sys.exit()   
            
    def get_non_global_rules(self):
        non_global = list(filter(lambda spec: spec["name"] != 'global' , self.specifications))
        names = list(map(lambda spec: spec['name'], non_global))
        return names
            
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
                val = self.stl_spec.get_value(i['name'])#[0]
                single_rho[i['name']] = val 
                total_rho += float(i['weight']) * val
                
        return total_rho, single_rho
    
    # fix of gemini TO BE TESTED
    def compute_robustness_dense(self, tau_state):
        
        # Prepara i dati di input
        data = {}
        for key in self.data.keys():
            data[key] = []
            
        # Riempie i dati dalle osservazioni
        for obs in tau_state:
            for i in self.stl_variables:
                if i['location'] == 'obs':
                    data[i['name']].append(obs[i['identifier']])
        
        # Genera la sequenza temporale (0, 1, 2, ...)
        data['time'] = list(range(len(tau_state)))
        
        single_rho = {}
        total_rho = 0.0
        
        if self.dense:
            # Valuta le specifiche
            self.stl_spec.evaluate(data)
            
            for i in self.specifications:
                raw_val = self.stl_spec.get_value(i['name'])
                
                # --- GESTIONE TIPO DI RITORNO (FIX IMPORTANTE) ---
                # RTAMT spesso restituisce una lista [[t0, val0], [t1, val1]...]
                if isinstance(raw_val, list):
                    # Prendiamo il valore al tempo 0 (inizio della finestra mobile)
                    # che riassume la robustezza per tutto l'orizzonte
                    val = float(raw_val[0][1])
                else:
                    # Caso in cui restituisca direttamente un float
                    val = float(raw_val)
                # -------------------------------------------------

                if val == 0.0:
                    print(f"DEBUG: Spec {i['name']} is 0.0. Weight: {i['weight']}")

                single_rho[i['name']] = val 
                total_rho += float(i['weight']) * val
                
        return total_rho, single_rho