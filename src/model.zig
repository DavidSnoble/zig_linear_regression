const std = @import("std");
const types = @import("types.zig");

pub fn predict(regression: types.Regression, x: f64) f64 {
    return regression.m * x + regression.b;
}

pub fn backwards_prop(predictions: []f64, dataset: types.DataSet) types.Derivative {
    const length = dataset.y.len;
    const n = @as(f64, @floatFromInt(length));
    var i: usize = 0;

    var df = std.heap.page_allocator.alloc(f64, length) catch unreachable;
    defer std.heap.page_allocator.free(df);
    for (predictions, dataset.y) |prediction, y| {
        df[i] = prediction - y;
        i += 1;
    }

    i = 0;
    var sum_dm: f64 = 0.0;
    for (df, dataset.x) |df_val, x| {
        sum_dm += (df_val * x);
        i += 1;
    }
    const dm = 2 * sum_dm / n;

    i = 0;
    var sum_db: f64 = 0.0;
    for (df) |df_val| {
        sum_db += df_val;
        i += 1;
    }
    const db = 2 * sum_db / n;

    return types.Derivative{ .dm = dm, .db = db };
}

pub fn cost(predictions: []f64, dataset: types.DataSet) f64 {
    const n = @as(f64, @floatFromInt(dataset.y.len));
    var sum: f64 = 0.0;
    for (predictions, dataset.y) |prediction, y| {
        sum += (prediction - y) * (prediction - y);
    }
    return sum / n;
}

pub fn train(dataset: types.DataSet, hyperparams: types.HyperParameters) types.Regression {
    const math_utils = @import("math_utils.zig");
    const length = dataset.y.len;
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rand = prng.random();

    var regression = types.Regression{
        .m = rand.float(f64) * 2.0 - 1.0,
        .b = rand.float(f64) * 2.0 - 1.0,
    };

    var current_lr = hyperparams.learning_rate;

    std.debug.print("\nStarting training with hyperparameters:\n", .{});
    std.debug.print("Experiment: {s}\n", .{hyperparams.name});
    std.debug.print("Epochs: {d}\n", .{hyperparams.epochs});
    std.debug.print("Initial learning rate: {d}\n", .{hyperparams.learning_rate});
    if (hyperparams.use_adaptive_lr) {
        std.debug.print("Using adaptive learning rate - Decay rate: {d}, Decay steps: {d}\n", .{ hyperparams.decay_rate, hyperparams.decay_steps });
    }
    math_utils.normalize(dataset.x);
    math_utils.normalize(dataset.y);

    var counter: usize = 0;
    while (counter <= hyperparams.epochs) {
        var predictions = std.heap.page_allocator.alloc(f64, length) catch unreachable;
        defer std.heap.page_allocator.free(predictions);
        var i: usize = 0;
        for (dataset.x) |x| {
            const prediction = predict(regression, x);
            predictions[i] = prediction;
            i += 1;
        }

        const derivative = backwards_prop(predictions, dataset);
        regression.m -= current_lr * derivative.dm;
        regression.b -= current_lr * derivative.db;

        if (hyperparams.use_adaptive_lr and counter % hyperparams.decay_steps == 0) {
            current_lr *= hyperparams.decay_rate;
        }

        const run_cost = cost(predictions, dataset);
        if (counter % 1000 == 0) {
            std.debug.print("Epoch: {d}, Cost: {d}\n", .{ counter, run_cost });
        }
        counter += 1;
    }
    return regression;
}

pub fn evaluate_model(regression: types.Regression, dataset: types.DataSet) !types.EvaluationResult {
    var predictions = std.ArrayList(f64).init(std.heap.page_allocator);

    for (dataset.x) |x| {
        const pred = predict(regression, x);
        try predictions.append(pred);
    }

    return types.EvaluationResult{
        .predictions = predictions,
    };
}

pub fn calculate_metrics(predictions: []const f64, dataset: types.DataSet) types.Metrics {
    var total_error: f64 = 0.0;

    for (predictions, dataset.y) |pred, y| {
        const test_error = pred - y;
        total_error += test_error * test_error;
    }

    const mse = total_error / @as(f64, @floatFromInt(dataset.y.len));
    const rmse = @sqrt(mse);

    return types.Metrics{
        .mse = mse,
        .rmse = rmse,
    };
}
