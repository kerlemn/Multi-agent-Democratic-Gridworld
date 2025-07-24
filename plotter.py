import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import tensorflow as tf

def animate_trajectory(traj,alives, grid_size, goals=None, interval=300, save_path=None):
    traj = [pos for pos in traj if pos is not None]

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid_size[0] - 0.5)
    ax.set_ylim(-0.5, grid_size[1] - 0.5)
    ax.set_xticks(range(grid_size[0]))
    ax.set_yticks(range(grid_size[1]))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title("trajectory")

    if goals:
        for agent_id, (gx, gy) in goals.items():
            ax.scatter(gx, gy, marker='*', s=200, color='red', label=f'Goal {agent_id}' if agent_id == 0 else None)

    path_line, = ax.plot([], [], lw=2, color='blue')
    point_marker, = ax.plot([], [], 'o', color='green')
    text = ax.text(0,0,"")

    def init():
        path_line.set_data([], [])
        point_marker.set_data([], [])
        text.set_text("")
        return path_line, point_marker, text

    def update(frame):
        x_vals = [x for x, y in traj[:frame+1]]
        y_vals = [y for x, y in traj[:frame+1]]
        path_line.set_data(x_vals, y_vals)
        point_marker.set_data([x_vals[-1]], [y_vals[-1]])
        text.set_text(str(alives[frame]))
        return path_line, point_marker, text

    ani = animation.FuncAnimation(fig, update, frames=len(traj),
                                  init_func=init, interval=interval, blit=True)

    plt.show()

def plot_value_functions(agents, width, height):
    num_agents = len(agents)
    max_z_layers = 2**num_agents

    fig, axes = plt.subplots(num_agents, max_z_layers, figsize=(4 * max_z_layers, 4 * num_agents), sharex=True, sharey=True)
    if num_agents == 1:
        axes = [axes]
    if max_z_layers == 1:
        axes = [[ax] for ax in axes]

    for row_idx, agent in enumerate(agents):
        z_values = {z: np.full((height, width), np.nan) for z in range(2**num_agents)}

        for (x, y, z), value in agent.V.items():
            if np.isnan(z_values[z][y, x]):
                z_values[z][y, x] = value
            else:
                z_values[z][y, x] += value

        minv = np.nanmin([np.nanmin(arr) for arr in z_values.values() if not np.isnan(arr).all()])
        maxv = np.nanmax([np.nanmax(arr) for arr in z_values.values() if not np.isnan(arr).all()])

        for col_idx, z in enumerate(sorted(z_values)):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(
                z_values[z],
                origin='lower',
                cmap='viridis',
                interpolation='none',
                vmin=minv,
                vmax=maxv
            )
            alive = format(z, f'0{num_agents}b')
            for i,(x,y) in enumerate(agent.goals.values()):
                if alive[i] == '1':
                    ax.scatter(x,y, marker='*', color = ('red' if i==row_idx else 'orange'))
            ax.set_title(f"A{agent.id}-{alive}")
    plt.tight_layout()
    plt.show()


def plot_Q_functions(agents, width, height):
    num_agents = len(agents)
    max_z_layers = 2**num_agents

    fig, axes = plt.subplots(num_agents, max_z_layers, figsize=(4 * max_z_layers, 4 * num_agents), sharex=True, sharey=True)
    if num_agents == 1:
        axes = [axes] 
    if max_z_layers == 1:
        axes = [[ax] for ax in axes]

    for row_idx, agent in enumerate(agents):
        z_values = {z: np.full((height, width), np.nan) for z in range(2**num_agents)}
        a_values = {z: np.full((height, width), '') for z in range(2**num_agents)}

        for (x, y, z, a), value in agent.Q.items():
            if np.isnan(z_values[z][y, x]) or value > z_values[z][y, x]:
                z_values[z][y, x] = value
                a_values[z][y, x] = a


        minv = np.nanmin([np.nanmin(arr) for arr in z_values.values() if not np.isnan(arr).all()])
        maxv = np.nanmax([np.nanmax(arr) for arr in z_values.values() if not np.isnan(arr).all()])

        for col_idx, z in enumerate(sorted(z_values)):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(
                z_values[z],
                origin='lower',
                cmap='viridis',
                interpolation='none',
                vmin=minv,
                vmax=maxv
            )
            alive = format(z, f'0{num_agents}b')
            for i,(x,y) in enumerate(agent.goals.values()):
                if alive[i] == '1':
                    ax.scatter(x,y, marker='*', color = ('red' if i==row_idx else 'orange'))
            ax.set_title(f"A{agent.id}-{alive}")
            for h in range(height):
                for w in range(width):
                    ax.text(w, h, a_values[col_idx][h,w], ha='center', va='center', fontsize=12, color='white')
    plt.tight_layout()
    plt.show()

def plot_episodes_optimality(steps, training = None, save = None):
    plt.plot(steps, label="optim")
    if training:
        plt.axvline(x=training, color='red', linestyle='--', label='Training/Evaluation')
    plt.legend()
    if save is not None:
        plt.savefig(f"./{save}.png")
        plt.clf()
    else:
        plt.show()


def plot_nn_weights(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights, biases = layer.get_weights()
            plt.figure(figsize=(10, 6))
            sns.heatmap(weights, cmap='coolwarm')
            plt.title(f'Weights Heatmap - Dense Layer {i}')
            plt.xlabel('Neurons')
            plt.ylabel('Inputs')
            plt.show()

def plot_nn_directions(agents, model_dict, width, height):

    action_to_arrow = {
        0: "↑", 1: "↓", 2: "←", 3: "→"
    }

    num_agents = len(agents)
    num_alive_states = 2 ** (num_agents - 1)

    fig, axes = plt.subplots(num_agents, num_alive_states, figsize=(4 * num_alive_states, 4 * num_agents))

    if num_agents == 1:
        axes = [axes]
    if num_alive_states == 1:
        axes = [[ax] for ax in axes]

    for agent_idx, agent in enumerate(agents):
        model = model_dict[agent.id]
        for z in range(num_alive_states):
            binary_alive = list(map(int, format(z, f'0{num_agents - 1}b')))

            ax = axes[agent_idx][z]
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(-0.5, height - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Agent {agent.id} - Aliveness {format(z, f'0{num_agents - 1}b')}")

            for i in range(width):
                for j in range(height):
                    onehot_x = [1 if k == i else 0 for k in range(width)]
                    onehot_y = [1 if k == j else 0 for k in range(height)]
                    input_vec = onehot_x + onehot_y + binary_alive + [1/len(agents)] * 4

                    input_tensor = tf.convert_to_tensor([input_vec], dtype=tf.float32)
                    q_values = model(input_tensor).numpy()[0]
                    best_action = int(np.argmax(q_values))
                    arrow = action_to_arrow[best_action]

                    ax.text(i, j, arrow, ha='center', va='center', fontsize=12)
            for i, (gx, gy) in enumerate(agent.goals.values()):
                ax.scatter(gx, gy, marker='*', s=150, color='red' if i == agent_idx else 'orange')

    plt.tight_layout()
    plt.show()

def computePareto(data):
        def is_dominated(p1, p2):
            return (p2[1] <= p1[1] and p2[2] <= p1[2]) and (p2[1] < p1[1] or p2[2] < p1[2])


        def pareto_sort(data):
            fronts = []
            remaining = data.copy()

            while remaining:
                    front = [p for p in remaining if not any(is_dominated(p, q) for q in remaining if q != p)]
                    fronts.append(front)
                    remaining = [p for p in remaining if p not in front]

            return fronts

        pareto_fronts = pareto_sort(data)
        colors = plt.colormaps.get_cmap("cool")

        plt.figure(figsize=(10, 7))

        for rank, front in enumerate(pareto_fronts, start=1):
                xs = [p[1] for p in front]
                ys = [p[2] for p in front]
                labels = [p[0] for p in front]

                color = colors(rank / max(1, len(pareto_fronts)))
                plt.scatter(xs, ys, s=80, color=color, label=f"Front {rank}")

                for x, y, label in zip(xs, ys, labels):
                        plt.text(x, y+0.1, f"{label}", fontsize=8)

                sorted_front = sorted(front, key=lambda x: (x[1], x[2]))
                fx, fy = zip(*[(p[1], p[2]) for p in sorted_front])
                plt.plot(fx, fy, color=color, linestyle='--', linewidth=1)

        plt.xlabel("(average) Optimality")
        plt.ylabel("% of errors")
        plt.title("Pareto Front Ranking")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./pareto.png")
        plt.clf()
     