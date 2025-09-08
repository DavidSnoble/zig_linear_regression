const std = @import("std");
const csv = @import("csv.zig");
const types = @import("types.zig");
const model = @import("model.zig");
const gui = @import("gui.zig");

// ExperimentResult is now defined in types.zig

pub fn evaluate_model(regression: types.Regression, dataset: types.DataSet) !types.EvaluationResult {
    var predictions = std.array_list.Managed(f64).init(std.heap.page_allocator);

    // Evaluate on dataset
    for (dataset.x) |x| {
        const pred = model.predict(regression, x);
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

pub fn run_experiment(train_dataset: types.DataSet, test_dataset: types.DataSet, hyperparams: types.HyperParameters) !types.ExperimentResult {
    const train_start: i64 = std.time.milliTimestamp();
    const train_result = try model.train(train_dataset, hyperparams);
    const train_end: i64 = std.time.milliTimestamp();

    const train_time = train_end - train_start;

    std.debug.print("Training completed in {d}ms\n", .{train_time});
    std.debug.print("Regression model: m = {d}, b = {d}\n", .{ train_result.regression.m, train_result.regression.b });

    const evaluation_result = try evaluate_model(train_result.regression, test_dataset);
    defer evaluation_result.predictions.deinit();

    const metrics = calculate_metrics(evaluation_result.predictions.items, test_dataset);

    std.debug.print("\nModel Performance Metrics for {s}:\n", .{hyperparams.name});
    std.debug.print("Mean Squared Error (MSE): {d:.6}\n", .{metrics.mse});
    std.debug.print("Root Mean Squared Error (RMSE): {d:.6}\n", .{metrics.rmse});

    return .{ .regression = train_result.regression, .history = train_result.history, .metrics = metrics };
}

pub fn main() !void {
    std.debug.print("Starting Regression Experiments!\n", .{});

    const filename = "data/data_for_lr.csv";
    const content = try csv.read_csv(filename);
    defer std.heap.page_allocator.free(content);

    const dataset = try parse_csv_data(content);
    defer std.heap.page_allocator.free(dataset.x);
    defer std.heap.page_allocator.free(dataset.y);

    const train_dataset = types.DataSet{ .x = dataset.x[0..500], .y = dataset.y[0..500] };
    const test_dataset = types.DataSet{ .x = dataset.x[500..], .y = dataset.y[500..] };

    // Define different hyperparameter configurations
    const experiments = [_]types.HyperParameters{
        .{ .epochs = 100000, .learning_rate = 0.00001, .name = "Fast Learning (Fixed LR)" },
        .{ .epochs = 100000, .learning_rate = 0.000005, .name = "Medium Learning (Fixed LR)" },
        .{
            .epochs = 100000,
            .learning_rate = 0.0001,
            .name = "Adaptive Learning Rate",
            .use_adaptive_lr = true,
            .decay_rate = 0.95,
            .decay_steps = 1000,
        },
    };

    // Run each experiment and collect results
    var experiment_results = try std.ArrayList(types.ExperimentResult).initCapacity(std.heap.page_allocator, experiments.len);
    defer {
        for (experiment_results.items) |*result| {
            result.history.losses.deinit(std.heap.page_allocator);
            result.history.epochs.deinit(std.heap.page_allocator);
        }
        experiment_results.deinit(std.heap.page_allocator);
    }

    for (experiments) |hyperparams| {
        const result = try run_experiment(train_dataset, test_dataset, hyperparams);
        experiment_results.appendAssumeCapacity(result);
    }

    // Launch GUI with training histories
    gui.run_gui_with_histories(experiment_results.items);
}

fn parse_csv_data(content: []u8) !types.DataSet {
    const allocator = std.heap.page_allocator;
    var x_list = std.array_list.Managed(f64).init(allocator);
    var y_list = std.array_list.Managed(f64).init(allocator);
    defer x_list.deinit();
    defer y_list.deinit();

    var iterator = std.mem.splitScalar(u8, content, '\n');
    var line_number: usize = 0;

    while (iterator.next()) |line| {
        line_number += 1;
        if (line.len == 0) continue;
        if (line[0] == 'x') continue;

        var data = std.mem.splitScalar(u8, line, ',');

        const x_str = data.next() orelse continue;
        const trimmed_x = std.mem.trim(u8, x_str, &std.ascii.whitespace);
        const x = std.fmt.parseFloat(f64, trimmed_x) catch |err| {
            std.debug.print("Error parsing x at line {d}: {s} - {}\n", .{ line_number, x_str, err });
            continue;
        };

        const y_str = data.next() orelse continue;
        // Trim the y value
        const trimmed_y = std.mem.trim(u8, y_str, &std.ascii.whitespace);
        const y = std.fmt.parseFloat(f64, trimmed_y) catch |err| {
            std.debug.print("Error parsing y at line {d}: '{s}' - {}\n", .{ line_number, trimmed_y, err });
            continue;
        };

        try x_list.append(x);
        try y_list.append(y);
    }
    return types.DataSet{ .x = try x_list.toOwnedSlice(), .y = try y_list.toOwnedSlice() };
}
