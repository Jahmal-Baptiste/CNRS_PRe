# coding: utf-8
"""
Balanced network of excitatory and inhibitory neurons.

An implementation of benchmarks 1 and 2 from

    Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398

The network is based on the CUBA and COBA models of Vogels & Abbott
(J. Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).


Usage: python VAbenchmarks.py [-h] [--plot-figure] [--use-views] [--use-assembly]
                              [--use-csa] [--debug DEBUG]
                              simulator benchmark protocol

positional arguments:
  simulator       neuron, nest, brian or another backend simulator
  benchmark       either CUBA or COBA
  protocol        either 1 or 2

optional arguments:
  -h, --help      show this help message and exit
  --plot-figure   plot the simulation results to a file
  --use-views     use population views in creating the network
  --use-assembly  use assemblies in creating the network
  --use-csa       use the Connection Set Algebra to define the connectivity
  --debug DEBUG   print debugging information


Andrew Davison, UNIC, CNRS
August 2006

"""

import socket
from math import *
from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging, normalized_filename
from pyNN.random import NumpyRNG, RandomDistribution
#import matplotlib
#matplotlib.use('Agg') #to have the right to plot on the simulation machine


# === Configure the simulator ================================================

sim, options = get_simulator(
                    ("benchmark", "either CUBA or COBA"),
                    ("protocol", "either 1 or 2"),
                    ("--plot-figure", "plot the simulation results to a file", {"action": "store_true"}),
                    ("--use-views", "use population views in creating the network", {"action": "store_true"}),
                    ("--use-assembly", "use assemblies in creating the network", {"action": "store_true"}),
                    ("--use-csa", "use the Connection Set Algebra to define the connectivity", {"action": "store_true"}),
                    ("--debug", "print debugging information"))

if options.use_csa:
    import csa

if options.debug:
    init_logging(None, debug=True)

timer = Timer()

# === Define parameters ========================================================

#protocol = "1"
threads  = 1
rngseed  = 98765
parallel_safe = True

n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.02  # connection probability
stim_dur = 30.   # (ms) duration of random stimulation
rate     = 100.  # (Hz) frequency of the random stimulation

dt       = 0.1   # (ms) simulation timestep
tstop = 1600 # (ms) simulaton duration #MODIFIED LINE
delay    = 0.2

# Cell parameters
area     = 20000. # (µm²)
tau_m    = 20.    # (ms)
cm       = 1.     # (µF/cm²)
g_leak   = 5e-5   # (S/cm²)
if options.benchmark == "COBA":
    E_leak   = -60.  # (mV)
elif options.benchmark == "CUBA":
    E_leak   = -49.  # (mV)
v_thresh = -50.   # (mV)
v_reset  = -60.   # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean   = -60.   # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc  = 5.     # (ms)
tau_inh  = 10.    # (ms)

# Synapse parameters
if options.benchmark == "COBA":
    Gexc = 4.     # (nS)
    Ginh = 51.    # (nS)
elif options.benchmark == "CUBA":
    Gexc = 0.27   # (nS) #Those weights should be similar to the COBA weights
    Ginh = 4.5    # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.     # (mV)
Erev_inh = -80.   # (mV)

### what is the synaptic delay???

# === Calculate derived parameters =============================================

area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells
n_inh = n - n_exc                     # number of inhibitory cells
if options.benchmark == "COBA":
    celltype = sim.IF_cond_exp
    w_exc    = Gexc*1e-3              # We convert conductances to uS
    w_inh    = Ginh*1e-3
elif options.benchmark == "CUBA":
    celltype = sim.IF_curr_exp
    w_exc = 1e-3*Gexc*(Erev_exc - v_mean)  # (nA) weight of excitatory synapses
    w_inh = 1e-3*Ginh*(Erev_inh - v_mean)  # (nA)
    assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

extra = {'threads' : threads,
         'filename': "va_%s.xml" % options.benchmark,
         'label': 'VA'}
if options.simulator == "neuroml":
    extra["file"] = "VAbenchmarks.xml"

node_id = sim.setup(timestep=dt, min_delay=delay, max_delay=1.0, **extra)
np = sim.num_processes()

host_name = socket.gethostname()
print("Host #%d is on %s" % (node_id + 1, host_name))

print("%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads']))

cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

if (options.benchmark == "COBA"):
    cell_params['e_rev_E'] = Erev_exc
    cell_params['e_rev_I'] = Erev_inh

timer.start()

print("%s Creating cell populations..." % node_id)
if options.use_views:
    # create a single population of neurons, and then use population views to define
    # excitatory and inhibitory sub-populations
    all_cells = sim.Population(n_exc + n_inh, celltype(**cell_params), label="All Cells")
    exc_cells = all_cells[:n_exc]
    exc_cells.label = "Excitatory cells"
    inh_cells = all_cells[n_exc:]
    inh_cells.label = "Inhibitory cells"
else:
    # create separate populations for excitatory and inhibitory neurons
    exc_cells = sim.Population(n_exc, celltype(**cell_params), label="Excitatory_Cells")
    inh_cells = sim.Population(n_inh, celltype(**cell_params), label="Inhibitory_Cells")
    if options.use_assembly:
        # group the populations into an assembly
        all_cells = exc_cells + inh_cells

print("%s Initialising membrane potential to random values..." % node_id)
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', low=v_reset, high=v_thresh, rng=rng)
if options.use_views:
    all_cells.initialize(v=uniformDistr)
else:
    exc_cells.initialize(v=uniformDistr)
    inh_cells.initialize(v=uniformDistr)


if options.benchmark == "COBA": #CHANGED SECTION
    stim_population = 5 #small population for high correlation of input
    t_stim1 = 0.
    ext_stim1 = sim.Population(stim_population, sim.SpikeSourcePoisson(start=t_stim1, rate=rate, duration=stim_dur), label="expoisson")
    if options.protocol  == "1":
        t_stim2 = 1000.
        ext_stim2 = sim.Population(stim_population, sim.SpikeSourcePoisson(start=t_stim2, rate=rate, duration=stim_dur), label="expoisson")    
    rconn = 0.01/stim_population #so that stim_population*rconn = 1 #used to be 0.01
    ext_conn = sim.FixedProbabilityConnector(rconn)
    ext_syn = sim.StaticSynapse(weight=0.1)


print("%s Connecting populations..." % node_id)
progress_bar = ProgressBar(width=20)
if options.use_csa:
    connector = sim.CSAConnector(csa.cset(csa.random(pconn)))
else:
    connector = sim.FixedProbabilityConnector(pconn, rng=rng, callback=progress_bar)
exc_syn = sim.StaticSynapse(weight=w_exc, delay=delay)
inh_syn = sim.StaticSynapse(weight=w_inh, delay=delay)

connections = {}
if options.use_views or options.use_assembly:
    connections['exc'] = sim.Projection(exc_cells, all_cells, connector, exc_syn, receptor_type='excitatory')
    connections['inh'] = sim.Projection(inh_cells, all_cells, connector, inh_syn, receptor_type='inhibitory')
    if (options.benchmark == "COBA"):
        connections['ext1'] = sim.Projection(ext_stim1, all_cells, ext_conn, ext_syn, receptor_type='excitatory')
else:
    connections['e2e'] = sim.Projection(exc_cells, exc_cells, connector, exc_syn, receptor_type='excitatory')
    connections['e2i'] = sim.Projection(exc_cells, inh_cells, connector, exc_syn, receptor_type='excitatory')
    connections['i2e'] = sim.Projection(inh_cells, exc_cells, connector, inh_syn, receptor_type='inhibitory')
    connections['i2i'] = sim.Projection(inh_cells, inh_cells, connector, inh_syn, receptor_type='inhibitory')
    if (options.benchmark == "COBA"):
        connections['ext12e'] = sim.Projection(ext_stim1, exc_cells, ext_conn, ext_syn, receptor_type='excitatory')
        connections['ext12i'] = sim.Projection(ext_stim1, inh_cells, ext_conn, ext_syn, receptor_type='excitatory')
        


# === Protocol conditions ==========================================================================

if options.protocol  == "2":
    #TO IMPLEMENT!!!
    stimulus = sim.ACSource(start=525.0, stop=1525.0, amplitude=0.1, frequency=4.0)

if (options.use_views or options.use_assembly) and options.benchmark == "COBA":
    if options.protocol  == "1":
        connections['ext2'] = sim.Projection(ext_stim2, all_cells, ext_conn, ext_syn, receptor_type='excitatory')
    elif options.protocol  == "2":
        #TO IMPLEMENT!!!
        all_cells.inject(stimulus)

elif (options.benchmark == "COBA"):
    if options.protocol  == "1":
        connections['ext22e'] = sim.Projection(ext_stim2, exc_cells, ext_conn, ext_syn, receptor_type='excitatory')
        connections['ext22i'] = sim.Projection(ext_stim2, inh_cells, ext_conn, ext_syn, receptor_type='excitatory')
    elif options.protocol  == "2":
        #TO IMPLEMENT!!!
        exc_cells.inject(stimulus)
        inh_cells.inject(stimulus)


# === Inject an extra current ======================================================================

print("%s Injecting current..." % node_id) #ADDED LINE

Chosen_injection = "none"

if Chosen_injection == "DC":
    pulse = sim.DCSource(amplitude=1., start=400.0, stop=600.0)
    #pulse.inject_into(exc_cells[0])
    exc_cells[0].inject(pulse)

    print("Getting the list of neurons connected to the injected neuron...") #ADDED LINE
    conn_indexes = {}
    if options.use_views or options.use_assembly:
        conn_indexes['02a'] = [0] #list of neurons connected to the neuron 0 post-synapticly, including it
        relevant_conn = connections['exc'].get('weight', format='list') #information on the connections between the excitatory and all the neurons
        for conn_tuple in relevant_conn:
            if conn_tuple[0] == 0.0 and conn_tuple[2] > 0:
                conn_indexes['02a'].append(conn_tuple[1])
        conn_indexes['02e'] = conn_indexes['02a'][:n_exc]
        conn_indexes['02i'] = conn_indexes['02a'][n_exc:]
    else:
        relevant_conn = {}
        conn_indexes['02e'] = [0] #list of excitatory neurons connected to the neuron 0 post-synapticly, including neuron 0
        conn_indexes['02i'] = [] #list of inhibitory neurons conected to the neuron 0 post-synapticly
        relevant_conn['e2e'] = connections['e2e'].get('weight', format='list')
        relevant_conn['e2i'] = connections['e2i'].get('weight', format='list')
        for conn_tuple in relevant_conn['e2e']:
            if conn_tuple[0] == 0.0 and conn_tuple[2] > 0:
                conn_indexes['02e'].append(conn_tuple[1])
        for conn_tuple in relevant_conn['e2i']:
            if conn_tuple[0] == 0 and conn_tuple[2] > 0:
                conn_indexes['02i'].append(conn_tuple[1])

    from numpy import array
    conn_indexes['02e'] = list(map(int, conn_indexes['02e']))
    conn_indexes['02i'] = list(map(int, conn_indexes['02i']))
    conn_indexes_02e = array(conn_indexes['02e'])
    conn_indexes_02i = array(conn_indexes['02i'])


if Chosen_injection == "Fake neurons":
    personal_progressbar = ProgressBar(width=20)
    fn_stim = sim.Population(50, sim.SpikeSourcePoisson(start=400, rate=100, duration=50), label="expoisson")
    rconn = 0.01
    fn_conn = sim.FixedProbabilityConnector(rconn, callback=personal_progressbar)
    fn_syn = sim.StaticSynapse(weight=0.1)
    if options.use_views or options.use_assembly:
        connections['fn'] = sim.Projection(fn_stim, all_cells, fn_conn, fn_syn, receptor_type='excitatory')
    else:
        connections['fn2e'] = sim.Projection(fn_stim, exc_cells, fn_conn, fn_syn, receptor_type='excitatory')
        connections['fn2i'] = sim.Projection(fn_stim, inh_cells, fn_conn, fn_syn, receptor_type='excitatory')





# === Setup recording ==========================================================
print("%s Setting up recording..." % node_id)
if options.use_views or options.use_assembly:
    all_cells.record('spikes')
    #exc_cells[[0, 1]].record('v') #ORIGINAL LINE
    #exc_cells.record('v') #CHANGED LINE
    #inh_cells.record('v') #ADDED LINE
    exc_cells[conn_indexes['02e']].record('v')
    inh_cells[conn_indexes['02i']].record('v') #ADDED LINE
else:
    exc_cells.record('spikes')
    inh_cells.record('spikes')
    #exc_cells[0, 1].record('v') #ORIGINAL LINE
    if Chosen_injection == "DC":
        exc_cells = exc_cells.__getitem__(conn_indexes_02e)
        inh_cells = inh_cells.__getitem__(conn_indexes_02i)
    exc_cells.record(('v', 'gsyn_exc')) #CHANGED LINE
    inh_cells.record(('v', 'gsyn_inh')) #ADDED LINE

buildCPUTime = timer.diff()

# === Save connections to file =================================================

#for prj in connections.keys():
    #connections[prj].saveConnections('Results/VAbenchmark_%s_%s_%s_np%d.conn' % (benchmark, prj, options.simulator, np))
saveCPUTime = timer.diff()

# === Run simulation ===========================================================

print("%d Running simulation..." % node_id)

sim.run(tstop)

simCPUTime = timer.diff()

E_count = exc_cells.mean_spike_count()
I_count = inh_cells.mean_spike_count()

# === Print results to file ====================================================

print("%d Writing data to file..." % node_id)

filename = normalized_filename("Results", "VAbenchmarks_%s_exc" % options.benchmark, "pkl",
                               options.simulator, np)
exc_cells.write_data(filename,
                     annotations={'script_name': __file__})
inh_cells.write_data(filename.replace("exc", "inh"),
                     annotations={'script_name': __file__})

writeCPUTime = timer.diff()

if options.use_views or options.use_assembly:
    connections = "%d e→e,i  %d i→e,i" % (connections['exc'].size(),
                                          connections['inh'].size())
elif Chosen_injection == "Fake neurons":
    connections = u"%d e→e  %d e→i  %d i→e  %d i→i  %d f→e  %d f→i" % (connections['e2e'].size(),
                                                       connections['e2i'].size(),
                                                       connections['i2e'].size(),
                                                       connections['i2i'].size(),
                                                       connections['fn2e'].size(),
                                                       connections['fn2i'].size())
else:
    connections = u"%d e→e  %d e→i  %d i→e  %d i→i" % (connections['e2e'].size(),
                                                       connections['e2i'].size(),
                                                       connections['i2e'].size(),
                                                       connections['i2i'].size())

if node_id == 0:
    print("\n--- Vogels-Abbott Network Simulation ---")
    print("Nodes                  : %d" % np)
    print("Simulation type        : %s" % options.benchmark)
    print("Number of Neurons      : %d" % n)
    print("Number of Synapses     : %s" % connections)
    print("Excitatory conductance : %g nS" % Gexc)
    print("Inhibitory conductance : %g nS" % Ginh)
    print("Excitatory rate        : %g Hz" % (E_count * 1000.0 / tstop,))
    print("Inhibitory rate        : %g Hz" % (I_count * 1000.0 / tstop,))
    print("Build time             : %g s" % buildCPUTime)
    #print("Save connections time  : %g s" % saveCPUTime)
    print("Simulation time        : %g s" % simCPUTime)
    print("Writing time           : %g s" % writeCPUTime)


# === Finished with simulator ==================================================

sim.end()