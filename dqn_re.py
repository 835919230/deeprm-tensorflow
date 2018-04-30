import DQN_Network
from pg_re import *

def get_dqn_traj(rl, env, episode_max_length):
    """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
    env.reset()
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    for i in range(episode_max_length):

        a = rl.choose_action(ob)

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob_, rew, done, info = env.step(a, repeat=True)

        rl.store_transition(ob, a, rew, ob_)

        if i % 100 == 0:
            rl.learn()

        rews.append(rew)

        if done:
            break

        ob = ob_
    rl.learn()
    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'info': info,
            }

def get_dqn_traj_worker(rl, env, pa):
    trajs = []

    for i in range(pa.num_seq_per_batch):
        traj = get_dqn_traj(rl, env, pa.episode_max_length)
        trajs.append(traj)

    # Compute discounted sums of rewards
    rets = [discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    return all_eprews, all_eplens, all_slowdown

def launch_dqn(pa, pg_resume=None, render = False, repre='image', end='no_new_job'):
    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    pg_learners = []
    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    for ex in range(pa.num_ex):
        print("-prepare for env-", ex)

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                              render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    print("-prepare for worker-")

    rl = DQN_Network.DeepQNetwork(n_actions=pa.network_output_dim,
                                  n_features=pa.network_input_height*pa.network_input_width,
                                  learning_rate=pa.lr_rate,
                                  reward_decay=0.9,
                                  e_greedy=0.9,
                                  replace_target_iter=200,
                                  memory_size=2000,
                                  )
    if pg_resume is not None:
        rl.load_data(pg_resume)

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------


    ref_discount_rews, ref_slow_down = slow_down_cdf.launch_with_env(pa=pa, env=envs[0], pg_resume=None, render=render,
                                                         plot=False, repre=repre, end=end)
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    num_epochs = pa.num_epochs + 1
    for iteration in range(1, num_epochs):

        ex_indices = list(range(pa.num_ex))
        np.random.shuffle(ex_indices)

        all_eprews = []
        eprews = []
        eplens = []
        all_slowdown = []

        eprewlist = []
        eplenlist =[]
        slowdownlist =[]
        losslist = []

        ex_counter = 0
        for ex in range(pa.num_ex):

            ex_idx = ex_indices[ex]

            eprew, eplen, slowdown = get_dqn_traj_worker(rl, envs[ex_idx], pa)
            eprewlist.append(eprew)
            eplenlist.append(eplen)
            slowdownlist.append(slowdown)

            ex_counter += 1

            if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

                print("\n\n")

                ex_counter = 0

                # all_eprews.extend([r["all_eprews"] for r in result])

                # eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                # eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

                # all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))

                # assemble gradients
                # grads = grads_all[0]
                # for i in range(1, len(grads_all)):
                # for j in range(len(grads)):
                # grads[j] += grads_all[i][j]

                # propagate network parameters to others
                # params = pg_learners[pa.batch_size].get_params()

                # rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

                # for i in range(pa.batch_size + 1):
                # pg_learners[i].set_net_params(params)


        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTrajs: \t %i" % len(eprewlist))
        print("NumTimesteps: \t %i" % np.sum(eplenlist))
        # print("MaxRew: \t %s" % np.average([np.max(rew) for rew in eprewlist]))
        # print("MeanRew: \t %s +- %s" % (np.mean(eprewlist), np.std(eprewlist)))
        print("MeanSlowdown: \t %s" % np.mean([np.mean(sd) for sd in slowdownlist]))
        print("MeanLen: \t %s +- %s" % (np.mean(eplenlist), np.std(eplenlist)))
        print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in eprewlist]))
        mean_rew_lr_curve.append(np.mean(eprewlist))
        slow_down_lr_curve.append(np.mean([np.mean(sd) for sd in slowdownlist]))

        if iteration % pa.output_freq == 0:

            rl.save_data(pa.output_filename + '_' + str(iteration))

            pa.unseen = True
            # slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.ckpt',
                                # render=False, plot=True, repre=repre, end=end)
            pa.unseen = False
            # test on unseen examples

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():
    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 50  # 1000
    pa.num_ex = 10  # 100
    pa.num_nw = 5
    pa.num_seq_per_batch = 10
    pa.output_freq = 50
    pa.batch_size = 10
    pa.output_filename = "dqn_data2_relu_under_zero/tmp"

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 20000  # 2000

    pa.num_epochs = 202
    pa.lr_rate = 0.02

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_3500.ckpt'

    render = False

    launch_dqn(pa, pg_resume, render, repre='image', end='all_done')

if __name__ == '__main__':
    main()
