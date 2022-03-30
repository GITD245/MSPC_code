import numpy as np
import fixed_env as env
import load_trace


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 2  # BB
CUSHION = 1  # BB
SUMMARY_DIR = './results'
LOG_FILE = '.results/mspc_'

PREDICT_STEP = 5
SWITCHPANELTY = 2.6*1e-5
SMOOTH_RATIO_TO_TARGET = 0.5
SLIDING_WINDOW_SIZE = 5
DURATION = 4

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# 
def quantization(c_bit_rate, available_bit_rate):
    max_level = len(available_bit_rate)-1
    min_bit_rate = available_bit_rate[0]
    max_bit_rate = available_bit_rate[-1]
    if c_bit_rate >= max_bit_rate:
        rate = max_level
    else:
        for i in range(max_level):
            if c_bit_rate < available_bit_rate[i+1]:
                rate = i
                break
    return rate

def multistep_pred(past_throughput, predict_step):
    future_throughput = np.zeros(predict_step)    #返回一个数组
    past_throughput_clone = [i for i in past_throughput]
    for i in range(predict_step):
        bandwidth_sum = 0
        nonzero_cnt = 0
        for j in range(len(past_throughput)):
            if past_throughput_clone[j] != 0:
                bandwidth_sum += 1.0/past_throughput_clone[j]
                nonzero_cnt += 1
        if nonzero_cnt == 0:
            pass
        else:
            future_throughput[i] = 1.0/(bandwidth_sum/nonzero_cnt)
        past_throughput_clone[0:-1] = past_throughput_clone[1:]
        past_throughput_clone[-1] = future_throughput[i]
    #print('past_throughput: {}'.format(past_throughput))
    return future_throughput

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    #log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_path = LOG_FILE + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    buffer_size = 0
    last_buffer_size = 0

    r_batch = []

    ############################################
    rate_set = []
    video_count = 0
    past_throughput = np.zeros(SLIDING_WINDOW_SIZE)
    k = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)
        #bit_rate = real_quality
        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) /      M_IN_K
        r_batch.append(reward)

        
        past_throughput[0:-1] = past_throughput[1:]
        past_throughput[-1] = float(video_chunk_size) / float(delay)*8 #kbps
        #print('past_throughput: {}'.format(past_throughput))
        #print('video_chunk_size : {}'.format(video_chunk_size))
        #print('delay : {}'.format(delay))
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(past_throughput[-1]) + '\t' + 
                       str(reward) + '\n')
        log_file.flush()


        ###################################################################
        ## Algorithm here
        ##
        ###################################################################
        switch_penalty_array = np.zeros((PREDICT_STEP, PREDICT_STEP))
        for i in range(PREDICT_STEP):
            switch_penalty_array[i,i] = SWITCHPANELTY * (PREDICT_STEP-i)

        # matrix initialization
        #缓存量预测所需计算G和F，第一项取决于当前的缓存量 ，第二部份由未来的码率决策所决定。
        matrix_E = np.zeros((PREDICT_STEP, PREDICT_STEP))
        matrix_F = matrix_E
        matrix_G = np.zeros((PREDICT_STEP, 2))
        for j in range(PREDICT_STEP):
            for i in range(j+1):
                matrix_E[j,i] = j-i+1
        for j in range(PREDICT_STEP):
            matrix_G[j,0] = j+2
            matrix_G[j,1] = -j-1

        #预测的带宽，multistep_pred
        estimated_throughput = multistep_pred(past_throughput, PREDICT_STEP)


        Bk = np.array([[buffer_size],[last_buffer_size]])

        TARGET_BUFFER = 8
        br = np.zeros((PREDICT_STEP+1,1))
        br[0,0] = buffer_size
        for i in range(PREDICT_STEP):
            br[i+1,0] = SMOOTH_RATIO_TO_TARGET*br[i,0] + (1-SMOOTH_RATIO_TO_TARGET)*TARGET_BUFFER
        # 预测的buffer
        target_buffer_array = br[1:]

        #F计算   -（L/estimated_throughput）
        matrix_F = matrix_E.dot(np.diag(-DURATION/estimated_throughput))

        #码率变化计算
        rate_change_array = np.linalg.inv(matrix_F.T.dot(matrix_F)+switch_penalty_array).dot(matrix_F.T).dot((target_buffer_array- matrix_G.dot(Bk)))
        #选择码率不超过target_rate
        target_rate = VIDEO_BIT_RATE[last_bit_rate] + rate_change_array[0]
        #quantizayion
        bit_rate = quantization(target_rate, VIDEO_BIT_RATE)

        '''
        if bit_rate - last_bit_rate > 1:
            bit_rate = last_bit_rate + 1
        elif bit_rate - last_bit_rate < -1:
            bit_rate = last_bit_rate - 1
        else:
            pass
        '''

        last_bit_rate = bit_rate
        last_buffer_size = buffer_size
        
        k += 1
        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            #print "video count", video_count
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = LOG_FILE + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()
