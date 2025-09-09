const std = @import("std");
const rl = @import("raylib");
const rg = @import("raygui");
const types = @import("types.zig");

// ExperimentResult is now defined in types.zig

/// `rl.getColor` only accepts a `u32`. Performing `@intCast` on the return value
/// of `rg.getStyle` invokes checked undefined behavior from Zig when passed to
/// `rl.getColor`, hence the custom implementation here...
fn getColor(hex: i32) rl.Color {
    var color: rl.Color = .black;
    // zig fmt: off
    color.r = @intCast((hex >> 24) & 0xFF);
    color.g = @intCast((hex >> 16) & 0xFF);
    color.b = @intCast((hex >>  8) & 0xFF);
    color.a = @intCast((hex >>  0) & 0xFF);
    // zig fmt: on
    return color;
}

pub fn run_gui() void {
    // Initialize raylib for rendering
    rl.setConfigFlags(rl.ConfigFlags{ .window_highdpi = true });
    rl.initWindow(400, 200, "Linear Regression Training Complete");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    var show_message_box = false;
    var should_exit = false;

    const color_int = rg.getStyle(.default, .{ .default = .background_color });

    while (!rl.windowShouldClose() and !should_exit) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(getColor(color_int));

        if (rg.button(.init(24, 24, 120, 30), "#191#Show Message"))
            show_message_box = true;

        if (rg.button(.init(160, 24, 120, 30), "#191#Exit"))
            should_exit = true;

        if (show_message_box) {
            const result = rg.messageBox(
                .init(85, 70, 250, 100),
                "#191#Training Complete",
                "Linear regression training finished successfully!",
                "Nice;Cool",
            );

            if (result >= 0) show_message_box = false;
        }
    }
}

pub fn run_gui_with_histories(histories: []const types.ExperimentResult) void {
    // Calculate optimal window size based on maximum epochs
    var max_epochs: usize = 0;
    for (histories) |exp| {
        for (exp.history.epochs.items) |epoch| {
            if (epoch > max_epochs) max_epochs = epoch;
        }
    }

    // Dynamic window sizing based on epoch count
    const base_width: i32 = 800;
    const base_height: i32 = 600;
    var window_width: i32 = base_width;
    var window_height: i32 = base_height;

    if (max_epochs > 50000) {
        window_width = @min(1600, base_width + @as(i32, @intCast(max_epochs / 10000)));
        window_height = @min(900, base_height + @as(i32, @intCast(max_epochs / 25000)));
    }

    // Initialize raylib for rendering with dynamic window size
    rl.setConfigFlags(rl.ConfigFlags{ .window_highdpi = true });
    rl.initWindow(@intCast(window_width), @intCast(window_height), "Linear Regression Training Results");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    var should_exit = false;
    var show_individual = false;
    var selected_experiment: usize = 0;

    const color_int = rg.getStyle(.default, .{ .default = .background_color });

    while (!rl.windowShouldClose() and !should_exit) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(getColor(color_int));

        // Draw graph area with dynamic sizing
        const graph_width = @as(f32, @floatFromInt(window_width - 100));
        const graph_height = @as(f32, @floatFromInt(window_height - 150));
        const graph_rect = rl.Rectangle{ .x = 50, .y = 50, .width = graph_width, .height = graph_height };
        rl.drawRectangleLinesEx(graph_rect, 2, rl.Color.black);

        // Draw graph title
        const title = if (show_individual) "Training Loss Over Time" else "Comparison: All Methods";
        rl.drawText(title, 300, 20, 20, rl.Color.black);

        if (histories.len > 0) {
            if (show_individual) {
                // Individual experiment view (original functionality)
                for (histories, 0..) |_, i| {
                    const btn_x = @as(f32, @floatFromInt(50 + @as(i32, @intCast(i)) * 120));
                    const exp_name = histories[i].history.experiment_name;
                    const btn_text = if (std.mem.eql(u8, exp_name, "Fast Learning (Fixed LR)"))
                        "#191#Fast"
                    else if (std.mem.eql(u8, exp_name, "Medium Learning (Fixed LR)"))
                        "#191#Medium"
                    else if (std.mem.eql(u8, exp_name, "Adaptive Learning Rate"))
                        "#191#Adaptive"
                    else
                        "#191#Exp";

                    if (rg.button(.init(btn_x, @as(f32, @floatFromInt(window_height - 80)), 100, 30), btn_text)) {
                        selected_experiment = @intCast(i);
                    }
                }

                draw_loss_graph(histories[selected_experiment].history, graph_rect);

                const metrics = histories[selected_experiment].metrics;
                const mse_text = std.fmt.allocPrint(std.heap.page_allocator, "Final MSE: {d:.6}", .{metrics.mse}) catch "Error";
                defer std.heap.page_allocator.free(mse_text);
                rl.drawText(@as([:0]const u8, @ptrCast(mse_text)), @as(i32, @intFromFloat(graph_rect.x + graph_rect.width - 200)), @as(i32, @intFromFloat(graph_rect.y + 10)), 16, rl.Color.black);
                const rmse_text = std.fmt.allocPrint(std.heap.page_allocator, "Final RMSE: {d:.6}", .{metrics.rmse}) catch "Error";
                defer std.heap.page_allocator.free(rmse_text);
                rl.drawText(@as([:0]const u8, @ptrCast(rmse_text)), @as(i32, @intFromFloat(graph_rect.x + graph_rect.width - 200)), @as(i32, @intFromFloat(graph_rect.y + 30)), 16, rl.Color.black);
            } else {
                // Comparison view - all experiments overlaid
                draw_comparison_graph(histories, graph_rect);

                // Display metrics for all experiments
                for (histories, 0..) |exp, i| {
                    const y_pos = @as(i32, @intFromFloat(graph_rect.y + graph_rect.height + 25 + @as(f32, @floatFromInt(i)) * 20));
                    const color = get_experiment_color(i);
                    const text = std.fmt.allocPrint(std.heap.page_allocator, "{s}: MSE={d:.4}", .{ exp.history.experiment_name, exp.metrics.mse }) catch "Error";
                    defer std.heap.page_allocator.free(text);
                    rl.drawText(@as([:0]const u8, @ptrCast(text)), 50, y_pos, 14, color);
                }

                // Add sampling indicator if any experiment uses downsampling
                var max_sampling_rate: usize = 1;
                for (histories) |exp| {
                    const rate = get_sampling_rate(exp.history.losses.items.len);
                    if (rate > max_sampling_rate) max_sampling_rate = rate;
                }
            }
        }

        // Toggle button between individual and comparison view
        const toggle_text = if (show_individual) "#191#Compare All" else "#191#Individual";
        const toggle_x = @as(f32, @floatFromInt(window_width - 250));
        if (rg.button(.init(toggle_x, @as(f32, @floatFromInt(window_height - 80)), 120, 30), toggle_text)) {
            show_individual = !show_individual;
        }

        // Exit button
        const exit_x = @as(f32, @floatFromInt(window_width - 120));
        if (rg.button(.init(exit_x, @as(f32, @floatFromInt(window_height - 80)), 100, 30), "#191#Exit"))
            should_exit = true;
    }
}

fn get_experiment_color(index: usize) rl.Color {
    return switch (index) {
        0 => rl.Color{ .r = 255, .g = 100, .b = 100, .a = 255 }, // Red
        1 => rl.Color{ .r = 100, .g = 100, .b = 255, .a = 255 }, // Blue
        2 => rl.Color{ .r = 100, .g = 255, .b = 100, .a = 255 }, // Green
        else => rl.Color{ .r = 255, .g = 100, .b = 255, .a = 255 }, // Magenta
    };
}

fn get_sampling_rate(total_points: usize) usize {
    if (total_points <= 10000) return 1; // Render all points for small datasets
    if (total_points <= 100000) return 10; // Render every 10th point for medium datasets
    if (total_points <= 1000000) return 100; // Render every 100th point for large datasets
    return 1000; // Render every 1000th point for very large datasets
}

fn get_sampled_indices(total_points: usize, sampling_rate: usize) []usize {
    const num_samples = (total_points + sampling_rate - 1) / sampling_rate; // Ceiling division
    var indices = std.heap.page_allocator.alloc(usize, num_samples) catch unreachable;

    var i: usize = 0;
    var sample_idx: usize = 0;
    while (i < total_points and sample_idx < num_samples) {
        indices[sample_idx] = i;
        i += sampling_rate;
        sample_idx += 1;
    }

    return indices;
}

fn draw_comparison_graph(histories: []const types.ExperimentResult, graph_rect: rl.Rectangle) void {
    if (histories.len == 0) return;

    // Calculate global min/max values across all experiments
    var global_min_loss = std.math.inf(f64);
    var global_max_loss = -std.math.inf(f64);
    var global_max_epoch: usize = 0;

    for (histories) |exp| {
        if (exp.history.losses.items.len == 0) continue;

        for (exp.history.losses.items) |loss| {
            if (loss < global_min_loss) global_min_loss = loss;
            if (loss > global_max_loss) global_max_loss = loss;
        }

        for (exp.history.epochs.items) |epoch| {
            if (epoch > global_max_epoch) global_max_epoch = epoch;
        }
    }

    if (global_max_loss == -std.math.inf(f64)) return;

    // Add some padding to the ranges
    const loss_range = global_max_loss - global_min_loss;
    const loss_padding = loss_range * 0.1;
    const min_loss = global_min_loss - loss_padding;
    const max_loss = global_max_loss + loss_padding;

    // Draw axes labels
    rl.drawText("Epochs", @intFromFloat(graph_rect.x + graph_rect.width / 2 - 30), @intFromFloat(graph_rect.y + graph_rect.height + 10), 12, rl.Color.black);
    rl.drawText("Loss", @intFromFloat(graph_rect.x - 40), @intFromFloat(graph_rect.y + graph_rect.height / 2 - 10), 12, rl.Color.black);

    // Draw grid lines and labels
    const num_grid_lines = 5;
    for (0..num_grid_lines) |i| {
        const fi = @as(f32, @floatFromInt(i));
        const x = graph_rect.x + (graph_rect.width / @as(f32, num_grid_lines)) * fi;
        const y = graph_rect.y + (graph_rect.height / @as(f32, num_grid_lines)) * fi;

        // Vertical grid lines (epochs)
        rl.drawLine(@intFromFloat(x), @intFromFloat(graph_rect.y), @intFromFloat(x), @intFromFloat(graph_rect.y + graph_rect.height), rl.Color.light_gray);
        const epoch_label = @as(usize, @intFromFloat(@as(f32, @floatFromInt(global_max_epoch)) * fi / @as(f32, num_grid_lines)));
        const epoch_text = std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{epoch_label}) catch "0";
        defer std.heap.page_allocator.free(epoch_text);
        rl.drawText(@as([:0]const u8, @ptrCast(epoch_text)), @intFromFloat(x - 10), @intFromFloat(graph_rect.y + graph_rect.height + 5), 10, rl.Color.black);

        // Horizontal grid lines (loss)
        rl.drawLine(@intFromFloat(graph_rect.x), @intFromFloat(y), @intFromFloat(graph_rect.x + graph_rect.width), @intFromFloat(y), rl.Color.light_gray);
        const loss_label = min_loss + (max_loss - min_loss) * (@as(f32, num_grid_lines) - fi) / @as(f32, num_grid_lines);
        const loss_text = std.fmt.allocPrint(std.heap.page_allocator, "{d:.4}", .{loss_label}) catch "0.0";
        defer std.heap.page_allocator.free(loss_text);
        rl.drawText(@as([:0]const u8, @ptrCast(loss_text)), @intFromFloat(graph_rect.x - 45), @intFromFloat(y - 5), 10, rl.Color.black);
    }

    // Draw legend
    for (histories, 0..) |exp, i| {
        const legend_y = @as(i32, @intFromFloat(graph_rect.y)) + 20 + @as(i32, @intCast(i)) * 20;
        const color = get_experiment_color(i);
        rl.drawRectangle(@intFromFloat(graph_rect.x + graph_rect.width - 150), legend_y - 5, 15, 15, color);
        const name_text = std.fmt.allocPrint(std.heap.page_allocator, "{s}", .{exp.history.experiment_name}) catch "Error";
        defer std.heap.page_allocator.free(name_text);
        rl.drawText(@as([:0]const u8, @ptrCast(name_text)), @intFromFloat(graph_rect.x + graph_rect.width - 130), legend_y, 12, rl.Color.black);
    }

    // Draw loss curves for each experiment with adaptive sampling
    for (histories, 0..) |exp, exp_index| {
        if (exp.history.losses.items.len == 0) continue;

        const color = get_experiment_color(exp_index);
        const sampling_rate = get_sampling_rate(exp.history.losses.items.len);
        const sampled_indices = get_sampled_indices(exp.history.losses.items.len, sampling_rate);
        defer std.heap.page_allocator.free(sampled_indices);

        // Draw sampled points and connect them
        for (1..sampled_indices.len) |sample_i| {
            const prev_idx = sampled_indices[sample_i - 1];
            const curr_idx = sampled_indices[sample_i];

            const prev_epoch = exp.history.epochs.items[prev_idx];
            const curr_epoch = exp.history.epochs.items[curr_idx];
            const prev_loss = exp.history.losses.items[prev_idx];
            const curr_loss = exp.history.losses.items[curr_idx];

            // Convert to screen coordinates using global ranges
            const prev_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(prev_epoch))) / @as(f32, @floatFromInt(global_max_epoch));
            const prev_y = graph_rect.y + graph_rect.height - (graph_rect.height * (prev_loss - min_loss)) / (max_loss - min_loss);
            const curr_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(curr_epoch))) / @as(f32, @floatFromInt(global_max_epoch));
            const curr_y = graph_rect.y + graph_rect.height - (graph_rect.height * (curr_loss - min_loss)) / (max_loss - min_loss);

            // Draw line segment
            rl.drawLine(@intFromFloat(prev_x), @intFromFloat(prev_y), @intFromFloat(curr_x), @intFromFloat(curr_y), color);

            // Draw small circle at each sampled point
            rl.drawCircle(@intFromFloat(curr_x), @intFromFloat(curr_y), 2, color);
        }

        // Draw first point
        if (sampled_indices.len > 0) {
            const first_idx = sampled_indices[0];
            const first_epoch = exp.history.epochs.items[first_idx];
            const first_loss = exp.history.losses.items[first_idx];
            const first_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(first_epoch))) / @as(f32, @floatFromInt(global_max_epoch));
            const first_y = graph_rect.y + graph_rect.height - (graph_rect.height * (first_loss - min_loss)) / (max_loss - min_loss);
            rl.drawCircle(@intFromFloat(first_x), @intFromFloat(first_y), 2, color);
        }
    }
}

fn draw_loss_graph(history: types.TrainingHistory, graph_rect: rl.Rectangle) void {
    if (history.losses.items.len == 0) return;

    // Find min/max values for scaling
    var min_loss = history.losses.items[0];
    var max_loss = history.losses.items[0];
    var max_epoch: usize = 0;

    for (history.losses.items, history.epochs.items) |loss, epoch| {
        if (loss < min_loss) min_loss = loss;
        if (loss > max_loss) max_loss = loss;
        if (epoch > max_epoch) max_epoch = epoch;
    }

    // Add some padding to the ranges
    const loss_range = max_loss - min_loss;
    const loss_padding = loss_range * 0.1;
    min_loss -= loss_padding;
    max_loss += loss_padding;

    // Draw axes labels
    rl.drawText("Epochs", @intFromFloat(graph_rect.x + graph_rect.width / 2 - 30), @intFromFloat(graph_rect.y + graph_rect.height + 10), 12, rl.Color.black);
    rl.drawText("Loss", @intFromFloat(graph_rect.x - 40), @intFromFloat(graph_rect.y + graph_rect.height / 2 - 10), 12, rl.Color.black);

    // Draw grid lines and labels
    const num_grid_lines = 5;
    for (0..num_grid_lines) |i| {
        const fi = @as(f32, @floatFromInt(i));
        const x = graph_rect.x + (graph_rect.width / @as(f32, num_grid_lines)) * fi;
        const y = graph_rect.y + (graph_rect.height / @as(f32, num_grid_lines)) * fi;

        // Vertical grid lines (epochs)
        rl.drawLine(@intFromFloat(x), @intFromFloat(graph_rect.y), @intFromFloat(x), @intFromFloat(graph_rect.y + graph_rect.height), rl.Color.light_gray);
        const epoch_label = @as(usize, @intFromFloat(@as(f32, @floatFromInt(max_epoch)) * fi / @as(f32, num_grid_lines)));
        const epoch_text = std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{epoch_label}) catch "0";
        defer std.heap.page_allocator.free(epoch_text);
        rl.drawText(@as([:0]const u8, @ptrCast(epoch_text)), @intFromFloat(x - 10), @intFromFloat(graph_rect.y + graph_rect.height + 5), 10, rl.Color.black);

        // Horizontal grid lines (loss)
        rl.drawLine(@intFromFloat(graph_rect.x), @intFromFloat(y), @intFromFloat(graph_rect.x + graph_rect.width), @intFromFloat(y), rl.Color.light_gray);
        const loss_label = min_loss + (max_loss - min_loss) * (@as(f32, num_grid_lines) - fi) / @as(f32, num_grid_lines);
        const loss_text = std.fmt.allocPrint(std.heap.page_allocator, "{d:.4}", .{loss_label}) catch "0.0";
        defer std.heap.page_allocator.free(loss_text);
        rl.drawText(@as([:0]const u8, @ptrCast(loss_text)), @intFromFloat(graph_rect.x - 45), @intFromFloat(y - 5), 10, rl.Color.black);
    }

    // Draw the loss curve with adaptive sampling
    const color = rl.Color{ .r = 255, .g = 100, .b = 100, .a = 255 }; // Red color for the line
    const sampling_rate = get_sampling_rate(history.losses.items.len);
    const sampled_indices = get_sampled_indices(history.losses.items.len, sampling_rate);
    defer std.heap.page_allocator.free(sampled_indices);

    // Draw sampled points and connect them
    for (1..sampled_indices.len) |sample_i| {
        const prev_idx = sampled_indices[sample_i - 1];
        const curr_idx = sampled_indices[sample_i];

        const prev_epoch = history.epochs.items[prev_idx];
        const curr_epoch = history.epochs.items[curr_idx];
        const prev_loss = history.losses.items[prev_idx];
        const curr_loss = history.losses.items[curr_idx];

        // Convert to screen coordinates
        const prev_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(prev_epoch))) / @as(f32, @floatFromInt(max_epoch));
        const prev_y = graph_rect.y + graph_rect.height - (graph_rect.height * (prev_loss - min_loss)) / (max_loss - min_loss);
        const curr_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(curr_epoch))) / @as(f32, @floatFromInt(max_epoch));
        const curr_y = graph_rect.y + graph_rect.height - (graph_rect.height * (curr_loss - min_loss)) / (max_loss - min_loss);

        // Draw line segment
        rl.drawLine(@intFromFloat(prev_x), @intFromFloat(prev_y), @intFromFloat(curr_x), @intFromFloat(curr_y), color);

        // Draw small circle at each sampled point
        rl.drawCircle(@intFromFloat(curr_x), @intFromFloat(curr_y), 2, color);
    }

    // Draw first point
    if (sampled_indices.len > 0) {
        const first_idx = sampled_indices[0];
        const first_epoch = history.epochs.items[first_idx];
        const first_loss = history.losses.items[first_idx];
        const first_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(first_epoch))) / @as(f32, @floatFromInt(max_epoch));
        const first_y = graph_rect.y + graph_rect.height - (graph_rect.height * (first_loss - min_loss)) / (max_loss - min_loss);
        rl.drawCircle(@intFromFloat(first_x), @intFromFloat(first_y), 2, color);
    }
}
