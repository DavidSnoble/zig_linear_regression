const std = @import("std");
const types = @import("types.zig");
const model = @import("model.zig");

// Thread-safe result collection
const ThreadSafeResults = struct {
    mutex: std.Thread.Mutex = .{},
    results: std.ArrayList(types.ExperimentResult),
    completed_count: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !ThreadSafeResults {
        return ThreadSafeResults{
            .results = try std.ArrayList(types.ExperimentResult).initCapacity(allocator, capacity),
        };
    }

    pub fn deinit(self: *ThreadSafeResults, allocator: std.mem.Allocator) void {
        self.results.deinit(allocator);
    }

    pub fn addResult(self: *ThreadSafeResults, allocator: std.mem.Allocator, result: types.ExperimentResult) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.results.append(allocator, result);
        _ = self.completed_count.fetchAdd(1, .monotonic);
    }

    pub fn getResults(self: *ThreadSafeResults) []types.ExperimentResult {
        return self.results.items;
    }

    pub fn isComplete(self: *const ThreadSafeResults, expected_count: usize) bool {
        return self.completed_count.load(.monotonic) == expected_count;
    }
};

// Experiment worker function that runs in its own thread
fn experimentWorker(
    train_dataset: types.DataSet,
    test_dataset: types.DataSet,
    hyperparams: types.HyperParameters,
    results: *ThreadSafeResults,
    allocator: std.mem.Allocator,
) void {
    // Each thread needs its own copy of the dataset to avoid data races
    // We'll copy the datasets for thread safety
    const train_x = allocator.dupe(f64, train_dataset.x) catch return;
    defer allocator.free(train_x);
    const train_y = allocator.dupe(f64, train_dataset.y) catch {
        allocator.free(train_x);
        return;
    };
    defer allocator.free(train_y);

    const test_x = allocator.dupe(f64, test_dataset.x) catch {
        allocator.free(train_x);
        allocator.free(train_y);
        return;
    };
    defer allocator.free(test_x);
    const test_y = allocator.dupe(f64, test_dataset.y) catch {
        allocator.free(train_x);
        allocator.free(train_y);
        allocator.free(test_x);
        return;
    };
    defer allocator.free(test_y);

    const thread_train_dataset = types.DataSet{ .x = train_x, .y = train_y };
    const thread_test_dataset = types.DataSet{ .x = test_x, .y = test_y };

    const result = runExperiment(thread_train_dataset, thread_test_dataset, hyperparams, allocator) catch |err| {
        std.debug.print("Experiment failed: {}\n", .{err});
        return;
    };

    results.addResult(allocator, result) catch |err| {
        std.debug.print("Failed to add result: {}\n", .{err});
        return;
    };
}

// Reimplementation of run_experiment for threading (to avoid circular imports)
fn runExperiment(
    train_dataset: types.DataSet,
    test_dataset: types.DataSet,
    hyperparams: types.HyperParameters,
    allocator: std.mem.Allocator,
) !types.ExperimentResult {
    const train_start: i64 = std.time.milliTimestamp();
    const train_result = try model.train(train_dataset, hyperparams);
    const train_end: i64 = std.time.milliTimestamp();

    const train_time = train_end - train_start;

    std.debug.print("Training completed in {d}ms\n", .{train_time});
    std.debug.print("Regression model: m = {d}, b = {d}\n", .{ train_result.regression.m, train_result.regression.b });

    var predictions = std.array_list.Managed(f64).init(allocator);
    defer predictions.deinit();

    // Evaluate on dataset
    for (test_dataset.x) |x| {
        const pred = model.predict(train_result.regression, x);
        try predictions.append(pred);
    }

    // Calculate metrics
    var total_error: f64 = 0.0;
    for (predictions.items, test_dataset.y) |pred, y| {
        const test_error = pred - y;
        total_error += test_error * test_error;
    }

    const mse = total_error / @as(f64, @floatFromInt(test_dataset.y.len));
    const rmse = @sqrt(mse);

    const metrics = types.Metrics{
        .mse = mse,
        .rmse = rmse,
    };

    std.debug.print("\nModel Performance Metrics for {s}:\n", .{hyperparams.name});
    std.debug.print("Mean Squared Error (MSE): {d:.6}\n", .{metrics.mse});
    std.debug.print("Root Mean Squared Error (RMSE): {d:.6}\n", .{metrics.rmse});

    return .{ .regression = train_result.regression, .history = train_result.history, .metrics = metrics };
}

// Function to run experiments in parallel
pub fn runExperimentsParallel(
    allocator: std.mem.Allocator,
    train_dataset: types.DataSet,
    test_dataset: types.DataSet,
    experiments: []const types.HyperParameters,
) ![]types.ExperimentResult {
    var results = try ThreadSafeResults.init(allocator, experiments.len);
    defer results.deinit(allocator);

    // Spawn threads for each experiment
    var threads = try allocator.alloc(std.Thread, experiments.len);
    defer allocator.free(threads);

    for (experiments, 0..) |hyperparams, i| {
        threads[i] = try std.Thread.spawn(.{}, experimentWorker, .{
            train_dataset,
            test_dataset,
            hyperparams,
            &results,
            allocator,
        });
    }

    // Wait for all threads to complete
    for (threads) |*thread| {
        thread.join();
    }

    // Return the collected results (copy them since results will be deinitialized)
    return allocator.dupe(types.ExperimentResult, results.getResults());
}

// Alternative implementation using thread pool pattern
pub fn runExperimentsParallelPool(
    allocator: std.mem.Allocator,
    train_dataset: types.DataSet,
    test_dataset: types.DataSet,
    experiments: []const types.HyperParameters,
) ![]types.ExperimentResult {
    // For simplicity, we'll use the same approach but could implement a more sophisticated pool
    return runExperimentsParallel(allocator, train_dataset, test_dataset, experiments);
}
