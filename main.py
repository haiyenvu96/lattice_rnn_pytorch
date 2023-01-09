from lattice import load_lattices, drawLattice, load_dictionary
from model import LatticeRNN
from trainer import Trainer


if __name__=="__main__":
    device = 'cuda'

    # Load lattices dataset
    lattices_data = load_lattices('./data/lattices.npz')
    ## Visualize one example
    # dot = drawLattice(lattices_data[10])
    # dot.render(directory='graph-outputs', view=True)

    # Load dictionary (tag2idx)
    tag2idx = load_dictionary('./data/dictionary.txt')
    for lattice in lattices_data:
        lattice.tags_idx = [tag2idx[tag] for tag in lattice.tags]

    # Split the dataset
    p_train = 0.9
    num_train = int(p_train * len(lattices_data))
    train_data, eval_data = lattices_data[:num_train], lattices_data[num_train:]
    print("Number of training data:", len(train_data))
    print("Number of eval data:", len(eval_data))
    print("===================================")

    # Define model
    lattice_rnn_model = LatticeRNN(input_dim=128, output_dim=1, hidden_dim=128)
    lattice_rnn_model.to(device)

    # Train the model
    N_epochs = 10
    n_logging_batches = 500
    checkpoint_dir = './checkpoints/'
    trainer = Trainer(
            lattice_rnn_model,
            train_data, eval_data,
            log_steps=n_logging_batches,
            epochs=N_epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
    trainer()

