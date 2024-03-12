
import numpy
from spiketools.sim.times import sim_spiketimes_poisson
import random
import quantities as pq
import elephant
import matplotlib.pyplot as plt
from spykesim import editsim

import numpy as np
import pandas as pd
from scipy import sparse

np.random.seed(4542)

# 10, 100
# 4,
#3 .33
#2 0
#1 0

plt.plot([1,2,3,4,5,6,7,8,9,10], [0,0,0.33,0,1.00,1.00,1.00,1.00,1.00,1.00])
plt.title("Repeats vs Acc")
plt.xlabel("Number of repeats")
plt.ylabel("Accuracy")
plt.show()

def euclidean_distance(mat1, mat2):
    dist = np.sqrt(np.sum(np.square(mat1 - mat2)))
    return dist

def genpoisson_spiketrain(rate, dt, duration):
    offset = duration
    events = np.cumsum(np.random.exponential(scale = 1 / rate, size = int(duration*rate + offset)))
    return np.round(events[np.logical_and(0 < events, events < duration)], -int(np.log10(dt)))
def genpoisson_spiketrains(nneurons, rate, dt, duration):
    spike_timings = np.array([], dtype = np.float)
    spike_neurons = np.array([], dtype = np.int)
    for n in range(nneurons):
        spike_train = genpoisson_spiketrain(rate, dt, duration)
        spike_timings = np.r_[spike_timings, spike_train]
        spike_neurons = np.r_[spike_neurons, n * np.ones_like(spike_train, dtype = np.int)]
    return pd.DataFrame({
        "neuronid": spike_neurons,
        "spiketime": spike_timings
    })

def gen_sequence(nneurons = 10, seqlen = 0.1, dt = 0.001):
    return np.round(np.linspace(dt, seqlen-dt, nneurons), int(-np.log10(dt)))

def gen_sequences(neurons = np.arange(10), nsequences = 10, start = 0, end = 60, seqlen = 0.01, dt = 0.001, shrink = 10):
    spike_timings = np.array([], dtype = np.float)
    spike_neurons = np.array([], dtype = np.int)
    nneurons = len(neurons)
    sequence_onsets = np.random.choice(
        np.arange(start, end - seqlen, seqlen),
        nsequences,
        replace = False
    )
    for onset in sequence_onsets:
        spike_timings = np.r_[spike_timings, onset + gen_sequence(nneurons, seqlen / shrink, dt)]
        spike_neurons = np.r_[spike_neurons, neurons]
    return pd.DataFrame({
        "neuronid": spike_neurons,
        "spiketime": spike_timings
    })

def gen_sequences_with_replay(shrinkages = [2], neurons = np.arange(10), nsequences = 10, duration = 60, seqlen = 0.1, dt = 0.001):
    duration_per_type = duration / (len(shrinkages) + 1)
    sequences = gen_sequences(neurons,
                              nsequences,
                              0,
                              duration_per_type,
                              seqlen,
                              dt)
    for idx, shrinkage in enumerate(shrinkages):
        replay = gen_sequences(neurons,
                               nsequences,
                               duration_per_type * (idx + 1),
                               duration_per_type * (idx + 2),
                               seqlen,
                               dt,
                               abs(shrinkage))
        if shrinkage < 0: # reverse replay
            replay = pd.DataFrame({
                "neuronid": replay.neuronid,
                "spiketime": np.copy(replay.spiketime[::-1])
            })
        sequences = pd.concat([sequences, replay])
    return sequences

def df2binarray_csc(df, duration_ms = 61, binwidth = 10):
    neuronids = df.neuronid
    print(neuronids)
    spikes_ms = df.spiketime * 1000
    nneurons = int(neuronids.max()+1)
    nrow = nneurons
    ncol = int(duration_ms) // binwidth + 1000
    binarray_lil = sparse.lil_matrix((nrow, ncol))
    print('here')
    print(int(neuronids.max()+1))
    for neuronid in range(nneurons):

        spike_train_of_a_neuron = spikes_ms[neuronids == neuronid]
        bins = np.arange(0, ncol * binwidth, binwidth)

        digitized_spike_train_of_a_neuron = np.digitize(spike_train_of_a_neuron, bins) - 1
        binned_spike_train_of_a_neuron = np.bincount(digitized_spike_train_of_a_neuron)
        binarray_lil[neuronid, digitized_spike_train_of_a_neuron] = binned_spike_train_of_a_neuron[digitized_spike_train_of_a_neuron]
    return binarray_lil.tocsc()

def gendata(rate):

    dt = 0.001
    # nsequences = 10
    # seqlen = 0.3
    nsequences = 1
    seqlen = 0.1
    shrinkages = [1,1]
    nneurons = 10
    duration = nsequences * seqlen * (len(shrinkages) + 1) + 0.2
    print('duration =', duration)
    nseqkinds = 2
    df = pd.DataFrame()
    df_sq = pd.DataFrame()
    for idx in range(nseqkinds):
        df_seq = gen_sequences_with_replay(
            shrinkages = shrinkages,
            neurons = np.arange(nneurons*(idx), nneurons*(idx+1)),
            nsequences = nsequences,
            duration = duration,
            seqlen = seqlen,
            dt = dt)
        df_seq = pd.DataFrame({
            "neuronid": df_seq.neuronid,
            "spiketime": np.copy(df_seq.spiketime + duration * idx + idx)
        })
        df = pd.concat([df, df_seq])
        df_sq = pd.concat([df_sq,df_seq])

    rate = rate
    nneurons = nneurons*nseqkinds
    duration = duration*nseqkinds + nseqkinds
    df_noise = genpoisson_spiketrains(nneurons, rate, dt, 30)
    df = df_noise
    print('running df2bin')
    print(int(duration*1000))
    binarray_csc = df2binarray_csc(df, duration_ms=int(duration*1000), binwidth = 100)
    return df, binarray_csc, df_sq


spike_times_poisson1 = sim_spiketimes_poisson(rate=5, duration=2)
spike_times_poisson2 = sim_spiketimes_poisson(rate=5, duration=2)



# Plot the spike times
#plot_rasters([spike_times_poisson1, spike_times_poisson2])
#plt.show()

xcorr = numpy.correlate(spike_times_poisson1, spike_times_poisson2, 'same')  # Compute the autocorrelation



spikes_total = []

sleep_spikes = []
awake_spikes = []

correlated_window_one = []
correlated_window_two = []

correlated_window_one = elephant.spike_train_generation.compound_poisson_process(t_start=0*pq.s, rate=5*pq.Hz, amplitude_distribution=[0]+[0.98]+[0]*8+[0.02], t_stop=10*pq.s)

for i in range(10):
    correlated_window_one.append(elephant.spike_train_generation.homogeneous_poisson_process(
        rate=5*pq.Hz, t_stop=10*pq.s))

    #plt.figure(figsize=(8, 3))
    #plt.eventplot([correlated_window_one[i].magnitude], linelengths=0.75, color='black')

#plt.show()



'''
patterns = elephant.spade.spade(
    spiketrains=correlated_window_one, binsize=1*pq.ms, winlen=1, min_spikes=3,
    n_surr=10,dither=5*pq.ms,
    psr_param=[0,0,0],
    output_format='patterns')['patterns']
'''
awake_rates = [14.927231045769622, 9.042558890585738, 5.825307148490364, 7.484126551273036, 8.088638408816077, 20.29252775743377, 12.662325524407773, 7.99942656291634, 26.262054707788682, 13.712334163453214, 13.905144946867733, 11.628647439091212, 2.1935144026896087, 16.928150977463908, 8.912690037939617, 2.6008989675412657, 9.115113406796212, 7.960981754970112, 5.63381061202097, 6.099819487231391, 12.155681446327534, 2.575121261510492, 7.862940439828116, 3.3390532697423794, 7.078744831605257, 8.68553949769409, 10.05429351357893, 26.50416279961166, 9.515428074881637, 2.2503367850088107, 10.96921903914817, 8.138894952193683, 7.337120313363157, 22.56194130913599, 20.649671773089292, 6.0717540030279595, 18.230910248814283, 29.88946987461271, 0.9672070756315625, 6.992066405191154, 5.643823078842837, 18.16147024988582, 10.128941042441076, 5.048022498541803, 10.300003436707192, 24.24313239015051, 7.679584795413281, 4.996740808444896, 1.88237590381719, 8.947828581569675, 6.041970876179965, 14.067575486337837, 9.576157963515898, 11.249032919386876, 20.217669130237944, 15.17550371333788, 7.879205202459857, 12.339094067724268, 17.96313450721721, 26.232098629806035, 4.749572880883161, 12.626924846993367, 8.135297915929296, 3.392011056750392, 6.495570016751809, 7.701526253889426, 6.996926739807533, 5.430557876306789, 18.030293414546946, 6.629526948981221, 8.820928306587755, 4.481766852795625, 1.6000197724332093, 7.20045242345489, 3.142881974333266, 4.070111774484679, 8.52573542077798, 11.01454357147398, 9.568692888896486, 10.766532063468926, 6.888218855046348, 9.999339897934368, 6.154106464338052, 14.95419376663179, 20.296006319628383, 15.742654668734948, 11.258390057313004, 6.616660572317137, 17.07557770563915, 1.762428100472674, 7.34510793189852, 0.7885352477591764, 4.084539874749772, 1.5058761471873388, 15.729413547612047, 10.987950862046317, 22.312948785458193, 9.79662827845883, 6.33804790165618, 3.715042057415196, 4.875546864586824, 4.689262364890875, 11.35425304077205, 15.381881168542844, 11.196530980441562, 5.243090741838742, 1.6988646877074842, 6.7656674394756156, 4.817590467815707, 4.653344875717991, 13.792878016421906, 4.222327537204521, 7.1731292383906515, 9.399796704826418, 6.052168504332473, 5.964530076809059, 11.005245456639463, 4.81505383495889, 6.291838686453572, 18.033869222612452, 4.111656211130295, 19.499518413987992, 2.270482815004185, 8.337249432056502, 7.265248296032707, 7.609030230005179, 2.6140437460943238, 11.68644667155574, 5.189671528441965, 23.748763485181392, 3.182589101359498, 11.715404422069893, 13.891608562504523, 21.41709853883657]

sleep_rates = [13.100841898082376, 6.88486884570428, 7.009497849684289, 6.072596478988138, 8.792510842639345, 11.634921031851297, 6.365169711214633, 6.395443780945018, 18.873021927123325, 9.639577368118283, 9.598578807849089, 14.945543756681973, 0.7744053270246437, 10.490202127329683, 3.227535424772004, 1.3272988756495068, 9.789008616715844, 3.557820209533458, 3.436850218698837, 6.529004539065701, 8.495560248259283, 3.95190495456696, 7.6394769354303, 0.4906723579568571, 7.9179949112278765, 6.829701727381593, 6.410195090242274, 22.80982621813004, 4.017922878136907, 0.39664098370769013, 12.66326941741275, 13.673533663761699, 8.148093430212453, 18.57739316053656, 12.354694898979453, 3.322075232311781, 11.988590365754147, 25.31128760122677, 4.968078289702678, 3.7902929794831057, 4.6561670234040715, 14.913601820024502, 6.7766436983460325, 3.4483086280201536, 8.280624298287673, 21.77510914165405, 12.44149275905493, 5.294266511302266, 4.08761841467666, 6.0305860941552885, 4.065795752293855, 14.104978502333783, 8.914335706851501, 10.65879120009911, 20.67861915832477, 12.628865904922177, 8.448782511764279, 14.03409023558632, 15.582191017893312, 13.7359860331741, 7.517336747120032, 5.118153760657423, 5.815648021637687, 6.286648078210777, 11.491238853276487, 1.49077927660098, 6.604281468263439, 0.3103829789244934, 7.8696273984990635, 5.840765246690506, 8.686652108320683, 5.802030703679646, 0.20953603885550343, 3.9813793948303315, 1.52751860998581, 0.5505528783566144, 6.13134813811677, 15.380359832944631, 8.064051935375991, 15.82314847687792, 8.11426276988075, 5.710593009424928, 5.033819146454938, 5.218639989483235, 14.068322537535398, 5.251281744173086, 10.419894855346657, 4.749731614161623, 9.413204667069571, 0.5167920597668083, 11.389905707880047, 0.8329081930749401, 7.862389210197906, 1.6867288891264172, 12.490242289511453, 6.202110672303206, 24.821254865062237, 5.979805164275733, 5.964508669585791, 0.9212278244215721, 5.736623490804803, 2.070852846358984, 9.012123655219815, 6.457357238611375, 12.503931778397193, 1.7011904910893816, 0.8091248940500851, 2.7432293774654752, 2.4806485489773054, 9.406882394719931, 5.645449115380338, 0.32545538847416117, 5.679273475751, 15.050770735028474, 1.7500443284735936, 8.856026984413306, 6.963471144856036, 12.468576892998646, 4.744320747266933, 12.566014719748441, 0.8424795993687061, 23.302379077366616, 3.015199579862415, 6.584385913972373, 2.698951413173504, 8.498374138653313, 2.5303486211799293, 4.9460946693588905, 10.513634452574236, 14.59409543707154, 0.5224679837499736, 8.086370075006938, 8.931751261408333, 6.396244629421165]
'''

print(len(awake_rates))

def generate_awake_spikes():
    print('here')
    magintude_vector = []
    correlated_window_one = elephant.spike_train_generation.compound_poisson_process(t_start=0 * pq.s, rate=5 * pq.Hz,
                                                                                     amplitude_distribution=[0] + [
                                                                                         0.999] + [0] * 8 + [0.001],
                                                                                     t_stop=10 * pq.s)

    line1 = 0
    line2 = 5
    delta = 0.01
    for i in range(134):
        spike_vector = elephant.spike_train_generation.homogeneous_poisson_process(
            rate=awake_rates[i] * pq.Hz, t_stop=10 * pq.s)
        if i > 40 and i < 80:
            if len(spike_vector) != 0:
                spike_vector[len(spike_vector) - 2] = 4
                spike_vector[len(spike_vector) - 1] = 9
                spike_vector[len(spike_vector) - 3] = line1
                line1 = line1 + delta
                spike_vector[len(spike_vector) - 4] = line2
                line2 = line2 + delta
                print('spike vector')
                print(spike_vector)

        correlated_window_one.append(spike_vector)
        magintude_vector.append(spike_vector.magnitude)
    return [magintude_vector, correlated_window_one]


[magintude_vector, correlated_window_one] = correlated_window_one = generate_awake_spikes()

plt.figure(figsize=(8, 3))
plt.eventplot(magintude_vector, linelengths=0.75, color='black')
plt.title("Awake Spikes");
plt.show()

def generate_sleep_spikes():
    print('here')
    magintude_vector = []
    ids = []
    correlated_window_one = elephant.spike_train_generation.compound_poisson_process(t_start=0 * pq.s, rate=5 * pq.Hz,
                                                                                     amplitude_distribution=[0] + [
                                                                          0.999] + [0] * 8 + [0.001],
                                                                                   t_stop=10 * pq.s)

    line1 = 0
    line2 = 5
    delta = 0.1
    line3 = 2
    line4 = 7
    for i in range(134):
        spike_vector = elephant.spike_train_generation.homogeneous_poisson_process(
            rate=sleep_rates[i] * pq.Hz, t_stop=10 * pq.s)
        if i > 40 and i < 80:
            print(len(spike_vector))
            if len(spike_vector) >=10:
                for j in range(len(spike_vector)):
                    ids.append(i-39)

                spike_vector[len(spike_vector)-2] = 1
                spike_vector[len(spike_vector)-1] = 1.1
                spike_vector[len(spike_vector) - 3] = 1.2
                line3 = line1 + delta
                line4 = line2 + delta
                line1 = line3 + delta

                spike_vector[len(spike_vector) - 4] = 1.3
                spike_vector[len(spike_vector) - 6] = 1.4
                spike_vector[len(spike_vector) - 5] = 1.5
                spike_vector[len(spike_vector) - 7] = 1.6
                spike_vector[len(spike_vector) - 8] = 1.7
                spike_vector[len(spike_vector) - 9] = 1.8
                spike_vector[len(spike_vector) - 10] = 1.9

                line2 = line2 + delta
                print('spike vector')
                print(spike_vector)

        correlated_window_one.append(spike_vector)
        magintude_vector.append(spike_vector.magnitude)
    return [magintude_vector, correlated_window_one, ids]


[magintude_vector, correlated_window_one, ids] = generate_sleep_spikes()

plt.figure(figsize=(8, 3))
plt.eventplot(magintude_vector[40:80], linelengths=0.75, color='black')
plt.title("Sleep Spikes");
plt.show()


patterns = elephant.spade.spade(
    spiketrains=correlated_window_one[40:80], binsize=1*pq.ms, winlen=1, min_spikes=3,
    n_surr=10,dither=5*pq.ms,
    psr_param=[0,0,0],
    output_format='patterns')['patterns']

viziphant.patterns.plot_patterns(correlated_window_one[40:80], patterns)
plt.show()

'''

def df2binarray_csc(df, duration_ms = 1000, binwidth = 1):
    print('running')
    neuronids = df.neuronid
    spikes_ms = df.spiketime * 1000
    nneurons = int(neuronids.max()+1)
    nrow = nneurons
    ncol = int(duration_ms) // binwidth + 1000
    print('cols')


    binarray_lil = sparse.lil_matrix((nrow, ncol))
    for neuronid in range(nneurons):
        spike_train_of_a_neuron = spikes_ms[neuronids == neuronid]
        bins = np.arange(0, ncol * binwidth, binwidth)
        print(len(bins))
        print(bins)
        digitized_spike_train_of_a_neuron = np.digitize(spike_train_of_a_neuron, bins) - 1
        binned_spike_train_of_a_neuron = np.bincount(digitized_spike_train_of_a_neuron)
        binarray_lil[neuronid, digitized_spike_train_of_a_neuron] = binned_spike_train_of_a_neuron[digitized_spike_train_of_a_neuron]
    return binarray_lil.tocsc()
'''
magintude_vector = magintude_vector[40:80]
df = pd.DataFrame(list(zip(ids, list(itertools.chain(*magintude_vector)))),
               columns =['neuronid', 'spiketime'])

#binarray_csc = df2binarray_csc(df, duration_ms=int(10 * 1000), binwidth=1)

'''

## 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
## 1 95 95 8 75 7 _ _ _ _ 7 15
a1 = range(1, 16)
a2 = [1, .95, .95, .8, .75, .7, .7, .7, .7, .7, .7, .7, .65, .65, .65]
plt.plot(a1, a2)
plt.ylim([0, 1])
plt.ylabel("Accuracy")
plt.xlabel("Background spiking rate (spikes/sec)")
plt.title("Accuracy as a function of background spiking rate")
plt.show()

df, binmat, dfs = gendata(15)
mat1 = binmat[:, 0:200].toarray()
mat2 = binmat[:, 200:400].toarray()


df_seq_final = pd.DataFrame()

for i in range(5):
    rand = random.random() * i
    rand_end_neuron = random.randint(9,18)
    spike_times = dfs.spiketime[10:15] * (i+1)*3
    neuron_ids = range(rand_end_neuron-5, rand_end_neuron)
    df_tmp = pd.DataFrame({
        "neuronid": neuron_ids,
        "spiketime": np.abs(spike_times)
    })
    df_seq_final = pd.concat([df_seq_final, df_tmp])

    t = spike_times[0] - 0.1
    spike_times = [t,t,t,t,t]
    neuron_ids = range(rand_end_neuron-5, rand_end_neuron)

    df_tmp = pd.DataFrame({
        "neuronid": neuron_ids,
        "spiketime": np.abs(spike_times)
    })
    df_seq_final = pd.concat([df_seq_final, df_tmp])


plt.plot(df.spiketime, df.neuronid, "k.")
plt.plot(df_seq_final.spiketime, df_seq_final.neuronid, "r.")

plt.title("Awake data")
plt.xlabel("Time(s)")
plt.ylabel("Neuron#")
plt.show()

arr = [0.01, 0]

df_seq_final = pd.DataFrame()
for j in range(30):
        if j%2==0:
            rand = random.uniform(((i+1)*.25), ((i+1)*0.26))
            rand_end_neuron = random.randint(9,18)

            spike_times = np.arange(0, 1, 0.05) * 0.25 + (j) #+ np.random.normal(0, 0.03, 20)

            print(spike_times)

            neuron_ids = range(0, 20)

            df_tmp = pd.DataFrame({
                "neuronid": neuron_ids,
                "spiketime": np.abs(spike_times)
            })

            df_seq_final = pd.concat([df_seq_final, df_tmp])

        else:
            rand = random.uniform(((i + 1) * .25), ((i + 1) * 0.26))
            rand_end_neuron = random.randint(9, 18)

            spike_times = np.arange(0, 1, 0.05) * 0.25 + (j)  # + np.random.normal(0, 0.2, 19)
            spike_times = np.flip(spike_times)

            print(spike_times)

            neuron_ids = range(0, 20)

            df_tmp = pd.DataFrame({
                "neuronid": neuron_ids,
                "spiketime": np.abs(spike_times)
            })

            df_seq_final = pd.concat([df_seq_final, df_tmp])



'''

    rand_end_neuron = random.randint(5, 18)
    spike_times = dfs.spiketime[10:15] * random.random()*10
    t = spike_times[0]  - 0.1 + random.random()*9
    spike_times = [t,t,t,t,t]
    neuron_ids = range(12, 17)

    df_tmp = pd.DataFrame({
        "neuronid": neuron_ids,
        "spiketime": spike_times
    })
    df_seq_final = pd.concat([df_seq_final, df_tmp])
    '''


plt.plot(df.spiketime, df.neuronid, "k.")
plt.plot(df_seq_final.spiketime, df_seq_final.neuronid, "r.")

plt.title("Awake data")
plt.xlabel("Time(s)")
plt.ylabel("Neuron#")
plt.show()

df = pd.concat([df, df_seq_final])
binmat = df2binarray_csc(df, 30000, 10)
print(df)
print('---------------')
print(binmat)

a = 0.5 #regularization term - pen term
es = editsim.FromBinMat(alpha=a)
#sim, _, _, _ = es._sim(mat1, mat2)

#print(f"The edit similarity between mat1 and mat2 is {sim}")


#more sparsity




##lines across all neurons and see what freq to detect
##multiple patterns to detect

window = 25 # ms

es.gensimmat(
    binmat, window=window, slide=10, numband=500, minhash=False
)

plt.imshow(es.simmat)
plt.show()


es.clustering(min_cluster_size=2)
es.gen_profile()
print("es profiles")
print(es.profiles)
my_dict = es.profiles




#print(np.array(es.profiles[list(my_dict.keys())[0]]).shape)
df = df.query('spiketime > 4.5')
df_seq_final = df_seq_final.query('spiketime > 4.5')
df = df.query('spiketime < 7')
df_seq_final = df_seq_final.query('spiketime < 7')

print(np.array(df.neuronid).shape)
print(np.array(df.spiketime).shape)
print(np.array(df_seq_final.neuronid).shape)
print(np.array(df_seq_final.spiketime).shape)
print(np.array(es.profiles[list(my_dict.keys())[0]]).shape)

final_df_neurons = pd.concat([df_seq_final.neuronid, df.neuronid]).to_numpy()
final_df_spikes = (round(pd.concat([df_seq_final.spiketime, df.spiketime]),2).to_numpy()-5)*100

plt.imshow(np.flip(es.profiles[list(my_dict.keys())[0]]))
#plt.gca().invert_yaxis()
#plt.scatter(final_df_spikes, final_df_neurons, s=1, color='white')
plt.show()

plt.imshow(np.flip(es.profiles[list(my_dict.keys())[1]]))
#plt.gca().invert_yaxis()
#plt.scatter(final_df_spikes, final_df_neurons, s=1, color='white')
plt.show()

#plt.imshow(np.flip(es.profiles[list(my_dict.keys())[2]]))
#plt.gca().invert_yaxis()
#plt.scatter(final_df_spikes, final_df_neurons, s=1, color='white')
#plt.show()


pf = es.profiles[list(my_dict.keys())[0]]

spikes = []
nid = []
tmp = []
nidt = []

for i in range(20):
    print('idx')
    idx = np.where(final_df_neurons == i)
    print(i)
    print(final_df_spikes[idx])
    #print(idx)
    #print(final_df_spikes[idx])
    #print('********************')
    for j in range(25):
        if(pf[i,j]>0.7):
            if len(final_df_spikes[idx]) != 0:
                print('here')
                #print(final_df_spikes[idx])
                print(i)
                print((5+j/100))
                print('--------------------')
                #if neuron at this time add

                if (5+j/100) in final_df_spikes[idx]:
                    spikes.append((5+j/100))
                    nid.append(i)

print('next')
print(final_df_neurons)
print('next')
print(final_df_spikes)

plt.plot(spikes, nid, "k.")
plt.xlim([5, 25])
plt.ylim([0, 20])
plt.show()



plt.plot(df.spiketime, df.neuronid, "k.")
plt.plot(df_seq_final.spiketime, df_seq_final.neuronid, "r.")
plt.title("Awake data")
plt.xlabel("Time(s)")
plt.ylabel("Neuron#")
plt.show()



'''
df, binmat = gendata(3)

plt.plot(df.spiketime, df.neuronid, "k.")
plt.title("Sleep data")
plt.xlabel("Time(s)")
plt.ylabel("Neuron#")
plt.show()
'''

es.clustering(min_cluster_size=2)

es.gen_profile()

sequence_dict = es.detect_sequences(cluster_id=0, th=20)
for i, (idx, mat) in enumerate(sequence_dict.items()):
    if i < 10:
        plt.imshow(mat, interpolation="nearest")
        print(euclidean_distance(es.profiles[0], mat))
        print(np.array(mat).shape)
        print(mat.shape)
        plt.title(f"Detected seq")
        plt.yticks(np.arange(0, 30, 5))
        plt.colorbar()
        plt.show()




plt.plot([0.01, 0.02, 0.03, 0.04, 0.05], [1, 1, 0.8, 0.3, 0])
plt.show()
