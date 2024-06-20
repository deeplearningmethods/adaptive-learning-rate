import numpy as np
import torch


def input_evaluation(model, test_x, use_test=True):
    if hasattr(model, 'test_loss') and use_test:
        test_loss = model.test_loss(test_x).cpu().detach().numpy()
    else:
        test_loss = model.loss(test_x).cpu().detach().numpy()

    result = test_loss

    return result


def evaluate(model, n_test, use_test=True, use_grad=True):
    test_x = model.initializer.sample(n_test)
    if use_grad:
        output = input_evaluation(model, test_x, use_test)
    else:
        with torch.no_grad():
            output = input_evaluation(model, test_x, use_test)

    return output


def reset_adam_find_lr(model, optim, test_steps, bs, n_test, nr_search,
                       lr_factor, init_lr, use_grad_eval=True):
    """Function which tests optimizer with nr_search different learning rates
     for test_steps steps and returns the learning rate which has lead to the
    smallest loss."""
    if nr_search > 1:
        torch.save({
            'model_state_dict': model.state_dict(),
            'adam_state_dict': optim.state_dict(),
        }, 'train_checkpoint.pt')
        lr_list = init_lr * lr_factor ** (np.arange(0.5 * (1 - nr_search),
                                                    0.5 * (1 + nr_search)))

        best_loss_test = 10000000.
        best_lr = init_lr
        for lr in lr_list:
            for g in optim.param_groups:
                g['lr'] = lr
            model.lr = lr
            for n in range(test_steps):
                optim.zero_grad()
                x = model.initializer.sample(bs)

                loss_value = model.loss(x)
                loss_value.backward()
                optim.step()
                model.done_steps += 1  # also count test steps for performance
            result = evaluate(model, n_test, False, use_grad_eval)

            test_value = result
            if test_value < best_loss_test:
                best_loss_test = test_value
                best_lr = lr

            checkpoint = torch.load('train_checkpoint.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['adam_state_dict'])

    else:
        best_lr = init_lr

    return best_lr


def reset_adam(model, nr_steps, bs, n_test, n_test_reset, nr_search, test_steps,
               lr_factor, init_lr=1e-3, eval_steps=50, tolerance=100,
               use_test_eval=True, eval_grad=True):
    """Variant of Adam where the learning rate is updated dynamically.
    After no improvement has occurred for tolerance many steps, we find a new
    learning rate. To do this, we start nr_search different runs with different
    learning rates from the current model for test_steps steps each,
    and then pick the rate which has lead to the smallest loss."""
    optim = torch.optim.Adam(model.parameters(), lr=init_lr)

    lrs = []
    errors = []
    train_losses = []

    step_list = []
    model.done_steps = 0

    current_lr = reset_adam_find_lr(model, optim, test_steps, bs, n_test_reset,
                                    nr_search, lr_factor, init_lr, use_grad_eval=eval_grad)
    if nr_search > 1:
        print(f'Best learning rate found: {current_lr}. Starting training.')
    for g in optim.param_groups:
            g['lr'] = current_lr
    model.lr = current_lr

    last_time_improved = 0
    best_loss = 10000000.
    for n in range(nr_steps):
        optim.zero_grad()

        x = model.initializer.sample(bs)
        loss_value = model.loss(x)
        loss_value.backward()
        optim.step()
        model.done_steps += 1
        loss = loss_value

        if loss < best_loss:
            best_loss = loss
            last_time_improved = 0

        if last_time_improved == tolerance:

            current_lr = reset_adam_find_lr(model, optim, test_steps, bs,
                                            n_test_reset, nr_search, lr_factor,
                                            current_lr, use_grad_eval=eval_grad)

            for g in optim.param_groups:
                g['lr'] = current_lr
            model.lr = current_lr
            last_time_improved = 1
            print(f'{n + 1} steps completed. New learning rate: {model.lr}')
        else:
            last_time_improved += 1

        if (n + 1) % eval_steps == 0:
            result = evaluate(model, n_test, use_test_eval, eval_grad)
            errors.append(result.item())
            lrs.append(model.lr)
            train_losses.append(loss.cpu().detach().numpy().item())
            print(f'{n+1} steps completed. Current training loss: {loss:.6f}')
            step_list.append(model.done_steps)

            if model.done_steps > nr_steps:
                break

    output = dict()
    output['lrs'] = lrs
    output['errors'] = errors
    output['train_losses'] = train_losses
    output['step_list'] = step_list

    return output


def test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test, n_test_reset,
                       search_list, k_list, tolerance, lr_factor,
                       use_test_eval=True, init_lr=1e-3):
    """ Tests standard Adam and reset-adam for k (number of tested learning
     rates) in k_list and searching steps in search_list. """
    results = dict()
    torch.save({'model_state_dict': ann.state_dict()}, 'model.pt')
    # saving model to reset parameters after each run
    print(f'Testing standard adam.')

    output = reset_adam(ann, train_steps, bs, n_test, 1, 1, 0, lr_factor,
                        eval_steps=eval_steps, tolerance=tolerance)

    test_errors = output['errors']
    lrs = output['lrs']
    train_errors = output['train_losses']
    eval_step_range = output['step_list']

    results['constlr'] = dict()

    results['constlr']['step_list'] = eval_step_range
    results['constlr']['errors'] = test_errors
    results['constlr']['train_loss'] = train_errors
    results['constlr']['lrs'] = lrs

    for layer in ann.layers:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    for search_steps in search_list:
        print(f'Using {search_steps} steps to determine new lr.')
        for k in k_list:
            checkpoint = torch.load('model.pt')
            ann.load_state_dict(checkpoint['model_state_dict'])
            print(f'Testing with {k} lr selections.')
            output = reset_adam(ann, train_steps, bs, n_test, n_test_reset, k,
                                search_steps, lr_factor, eval_steps=eval_steps,
                                tolerance=tolerance, use_test_eval=use_test_eval,
                                init_lr=init_lr)

            save_string = f'k{k}ss{search_steps}'

            test_errors = output['errors']
            lrs = output['lrs']
            train_errors = output['train_losses']
            eval_step_range = output['step_list']

            results[save_string] = dict()

            results[save_string]['step_list'] = eval_step_range
            results[save_string]['errors'] = test_errors
            results[save_string]['train_loss'] = train_errors
            results[save_string]['lrs'] = lrs

    return results