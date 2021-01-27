import sys
sys.path.append("/home/caor/Documents/gym-mil")
sys.path.append("/home/caor/Documents/mujoco-py-0.5.7")
import gym
env = gym.make('ReacherMILTest-v1')

class EvalMilReach(object):

    def __init__(self, dataset, num_subtasks, num_trials, render=True):
        self.env = gym.make('ReacherMILTest-v1')
        self.env.env.set_visibility(render)
        self.demos = dataset.data

    def get_embedding(self, task_index, demo_indexes, emb_mod):
        image_files = [
            self.demos[task_index][j]['image_files'] for j in demo_indexes]
        states = [
            self.demos[task_index][j]['states'] for j in demo_indexes]
        outs = [
            self.demos[task_index][j]['actions'] for j in demo_indexes]

        feed_dict = {
            self.outputs['input_image_files']: image_files,
            self.outputs['input_states']: states,
            self.outputs['input_outputs']: outs,
        }
        embedding, = self.sess.run(
            self.outputs['sentences'], feed_dict=feed_dict)
 
        return emb_mod.forward(inputs, eval=True)['support_embeddings']

    def get_action(self, obs, state, embedding):
        feed_dict = {
            self.outputs['ctrnet_images']: [[obs]],
            self.outputs['ctrnet_states']: [[state]],
            self.outputs['sentences']: [embedding],
        }
        action, = self.sess.run(
            self.outputs['output_actions'], feed_dict=feed_dict)
        return action

    def evaluate(self, iter, emb_mod, ctr_mod):

        print("Evaluating at iteration: %i" % iter)
        # iter_dir = os.path.join(self.record_gifs_dir, 'iter_%i' % iter)
        # utils.create_dir(iter_dir)
        self.env.reset()

        successes = []
        for i in range(self.num_subtasks):

            # TODO hacked in for now. Remove 0
            dem_conds = self.demos[i][0]['demo_selection']

            # randomly select a demo from each of the folders
            selected_demo_indexs = random.sample(
                range(len(self.demos[i])), self.supports)

            embedding = self.get_embedding(i, selected_demo_indexs)
            # gifs_dir = self.create_gif_dir(iter_dir, i)

            for j in range(self.num_trials):
                if j in dem_conds:
                    distances = []
                    observations = []
                    for t in range(self.time_horizon):
                        self.env.render()
                        # Observation is shape (64,80,3)
                        obs, state = self.env.env.get_current_image_obs()
                        observations.append(obs)
                        obs = ((obs / 255.0) * 2.) - 1.

                        action = self.get_action(obs, state, embedding)
                        ob, reward, done, reward_dict = self.env.step(
                            np.squeeze(action))
                        dist = -reward_dict['reward_dist']
                        if t >= self.time_horizon - REACH_SUCCESS_TIME:
                            distances.append(dist)
                    if np.amin(distances) <= REACH_SUCCESS_THRESH:
                        successes.append(1.)
                    else:
                        successes.append(0.)
                    # self.save_gifs(observations, gifs_dir, j)

                self.env.render(close=True)
                self.env.env.next()
                self.env.env.set_visibility(self.render)
                self.env.render()

        self.env.render(close=True)
        self.env.env.reset_iter()
        final_suc = np.mean(successes)
        print("Final success rate is %.5f" % final_suc)
        return final_suc