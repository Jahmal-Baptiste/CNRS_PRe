
	# LFP
	# 95% of the LFP signal is a result of all exc and inh cells conductances from 250um radius from the tip of the electrode (Katzner et al. 2009).
	# Therefore we include all recorded cells but account for the distance-dependent contribution weighting currents /r^2
	# We assume that the electrode has been placed in the cortical coordinates <tip>
	# Only excitatory neurons are relevant for the LFP (because of their geometry) Bartosz
	filtered_dsv = param_filter_query(data_store, sheet_name=lfp_sheet, st_name=stimulus)
	filtered_segs = filtered_dsv.get_segments()
	for s in filtered_segs:
		lfp_neurons = filtered_segs[0].get_stored_vm_ids()
	print lfp_neurons

	if lfp_neurons == None:
		print "No Exc Vm recorded.\n"
		return
	print "Recorded neurons for LFP:", len(lfp_neurons)
	lfp_positions = data_store.get_neuron_postions()[lfp_sheet] # position is in visual space degrees

	# choose LFP tip position
	# select all neurons id having a certain orientation preference
	or_neurons = lfp_neurons
	# if lfp_sheet=='V1_Exc_L4':
	# 	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	# 	exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=lfp_sheet)
	# 	print "post - LGNAfferentOrientation", exc_or
	# 	if len(exc_or):
	# 		if lfp_opposite:
	# 			exc_or_g = numpy.array(lfp_neurons)[numpy.nonzero(numpy.array([circular_dist(exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in lfp_neurons]) < .1)[0]]
	# 		else:
	# 			exc_or_g = numpy.array(lfp_neurons)[numpy.nonzero(numpy.array([circular_dist(exc_or.get_value_by_id(i),0,numpy.pi)  for i in lfp_neurons]) < .1)[0]]
	# 	or_neurons = list(exc_or_g)

	or_sheet_ids = data_store.get_sheet_indexes(sheet_name=lfp_sheet, neuron_ids=or_neurons)
	or_neurons = select_ids_by_position(lfp_positions, or_sheet_ids, radius=radius)
	print "Oriented neurons idds to choose the LFP tip electrode location:", len(or_neurons), or_neurons

	xs = lfp_positions[0][or_neurons]
	ys = lfp_positions[1][or_neurons]
	# create list for each point
	or_dists = [[] for i in range(len(or_neurons))]
	selected_or_ids = [[] for i in range(len(or_neurons))]
	for i,_ in enumerate(or_neurons):
		# calculate distance from all others
		for j,o in enumerate(or_neurons):
			dist = math.sqrt( (xs[j]-xs[i])**2 + (ys[j]-ys[i])**2 )
			if dist <= 0.8: # minimal distance between oriented blots in cortex 
				selected_or_ids[i].append(o)
				# print lfp_positions[0][o], lfp_positions[1][o]
	# pick the largest list	
	selected_or_ids.sort(key = lambda x: len(x), reverse=True)
	print selected_or_ids[0]
	# average the selected xs and ys to generate the centroid, which is the tip
	x = 0
	y = 0
	for i in selected_or_ids[0]:
		# print i
		x = x + lfp_positions[0][i]
		y = y + lfp_positions[1][i]
	x = x / len(selected_or_ids[0])
	y = y / len(selected_or_ids[0])
	tip = [[x],[y],[.0]]
	print "LFP electrod tip location (x,y) in degrees:", tip

	# not all neurons are necessary, >100 are enough
	chosen_ids = numpy.random.randint(0, len(lfp_neurons), size=100 )
	# print chosen_ids
	lfp_neurons = lfp_neurons[chosen_ids]

	distances = [] # distances form origin of all excitatory neurons
	sheet_e_ids = data_store.get_sheet_indexes(sheet_name=lfp_sheet, neuron_ids=lfp_neurons) # all neurons
	magnification = 1000 # magnification factor to convert the degrees in to um
	if "X" in lfp_sheet:
		magnification = 200
	for i in sheet_e_ids:
		distances.append( numpy.linalg.norm( numpy.array((lfp_positions[0][i],lfp_positions[1][i],lfp_positions[2][i])) - numpy.array(tip) ) *magnification ) 
	distances = numpy.array(distances)
	print "Recorded distances:", len(distances)#, distances #, distances**2

	# gather vm and conductances
	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=lfp_sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)
	ticks = set([])
	for x in segs:
		ticks.add( getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) )
	ticks = sorted(ticks)
	num_ticks = len( ticks )
	print ticks
	trials = len(segs) / num_ticks
	print "trials:",trials
	gc.collect()

	pop_vm = []
	pop_gsyn_e = []
	pop_gsyn_i = []
	for n,idd in enumerate(lfp_neurons):
		print "idd", idd
		full_vm = [s.get_vm(idd) for s in segs]
		full_gsyn_es = [s.get_esyn(idd) for s in segs]
		full_gsyn_is = [s.get_isyn(idd) for s in segs]
		# print "len full_gsyn_e/i", len(full_gsyn_es) # 61 = 1 spontaneous + 6 trial * 10 num_ticks
		# print "shape gsyn_e/i", full_gsyn_es[0].shape
		# mean input over trials
		mean_full_vm = numpy.zeros((num_ticks, full_vm[0].shape[0])) # init
		mean_full_gsyn_e = numpy.zeros((num_ticks, full_gsyn_es[0].shape[0])) # init
		mean_full_gsyn_i = numpy.zeros((num_ticks, full_gsyn_es[0].shape[0]))
		# print "shape mean_full_gsyn_e/i", mean_full_gsyn_e.shape
		sampling_period = full_gsyn_es[0].sampling_period
		t_stop = float(full_gsyn_es[0].t_stop - sampling_period) # 200.0
		t_start = float(full_gsyn_es[0].t_start)
		time_axis = numpy.arange(0, len(full_gsyn_es[0]), 1) / float(len(full_gsyn_es[0])) * abs(t_start-t_stop) + t_start
		# sum by size
		t = 0
		for v,e,i in zip(full_vm, full_gsyn_es, full_gsyn_is):
			s = int(t/trials)
			v = v.rescale(mozaik.tools.units.mV) 
			e = e.rescale(mozaik.tools.units.nS) # NEST is in nS, PyNN is in uS
			i = i.rescale(mozaik.tools.units.nS) # NEST is in nS, PyNN is in uS
			mean_full_vm[s] = mean_full_vm[s] + numpy.array(v.tolist())
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] + numpy.array(e.tolist())
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] + numpy.array(i.tolist())
			t = t+1

		# average by trials
		for s in range(num_ticks):
			mean_full_vm[s] = mean_full_vm[s] / trials
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] / trials
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] / trials

		pop_vm.append(mean_full_vm)
		pop_gsyn_e.append(mean_full_gsyn_e)
		pop_gsyn_i.append(mean_full_gsyn_i)

	pop_v = numpy.array(pop_vm)
	pop_e = numpy.array(pop_gsyn_e)
	pop_i = numpy.array(pop_gsyn_i)

	# We produce the current for each cell for this time interval, with the Ohm law:
	# I = ge(V-Ee) + gi(V+Ei)
	# where 
	# Ee is the equilibrium for exc, which is 0.0
	# Ei is the equilibrium for inh, which is -80.0
	i = pop_e*pop_v + pop_i*(pop_v-80.0)
	# the LFP is the result of cells' currents
	avg_i = numpy.average( i, weights=distances**2, axis=0 )
	std_i = numpy.std( i, axis=0 )
	sigma = 0.1 # [0.1, 0.01] # Dobiszewski_et_al2012.pdf
	lfp = ( (1/(4*numpy.pi*sigma)) * avg_i ) / std_i # Z-score
	print "LFP:", lfp.shape, lfp.min(), lfp.max()
	print lfp

	#TEST: plot the LFP for each stimulus
	for s in range(num_ticks):
		# for each stimulus plot the average conductance per cell over time
		matplotlib.rcParams.update({'font.size':22})
		fig,ax = plt.subplots()

		ax.plot( range(0,len(lfp[s])), lfp[s], color=color, linewidth=3 )

		ax.set_ylim([lfp.min(), lfp.max()])
		ax.set_ylabel( "LFP (z-score)" )
		ax.set_xlabel( "Time (ms)" )

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.xaxis.set_ticks(ticks, ticks)
		ax.yaxis.set_ticks_position('left')

		# text
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseLFP_"+lfp_sheet+"_"+spike_sheet+"_"+parameter+"_"+str(ticks[s])+"_"+addon+".svg", dpi=200, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()