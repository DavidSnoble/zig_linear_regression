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
    // Initialize raylib for rendering with larger window for graphs
    rl.initWindow(800, 600, "Linear Regression Training Results");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    var should_exit = false;
    var selected_experiment: usize = 0;

    const color_int = rg.getStyle(.default, .{ .default = .background_color });

    while (!rl.windowShouldClose() and !should_exit) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(getColor(color_int));

        // Draw graph area
        const graph_rect = rl.Rectangle{ .x = 50, .y = 50, .width = 700, .height = 450 };
        rl.drawRectangleLinesEx(graph_rect, 2, rl.Color.black);

        // Draw graph title
        rl.drawText("Training Loss Over Time", 300, 20, 20, rl.Color.black);

        // Draw experiment selector
        if (histories.len > 0) {
            for (histories, 0..) |_, i| {
                const btn_x = @as(f32, @floatFromInt(50 + @as(i32, @intCast(i)) * 120));
                // Use a simple approach - just show abbreviated names for buttons
                const exp_name = histories[i].history.experiment_name;
                const btn_text = if (std.mem.eql(u8, exp_name, "Fast Learning (Fixed LR)"))
                    "#191#Fast"
                else if (std.mem.eql(u8, exp_name, "Medium Learning (Fixed LR)"))
                    "#191#Medium"
                else if (std.mem.eql(u8, exp_name, "Adaptive Learning Rate"))
                    "#191#Adaptive"
                else
                    "#191#Exp";

                if (rg.button(.init(btn_x, 520, 100, 30), btn_text)) {
                    selected_experiment = @intCast(i);
                }
            }

            // Draw the loss graph for selected experiment
            draw_loss_graph(histories[selected_experiment].history, graph_rect);

            // Display final metrics (moved below buttons to avoid overlap)
            const metrics = histories[selected_experiment].metrics;
            rl.drawText(rl.textFormat("Final MSE: %.6f", .{metrics.mse}), 50, 570, 16, rl.Color.black);
            rl.drawText(rl.textFormat("Final RMSE: %.6f", .{metrics.rmse}), 50, 590, 16, rl.Color.black);
        }

        // Exit button
        if (rg.button(.init(650, 520, 100, 30), "#191#Exit"))
            should_exit = true;
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
        rl.drawText(rl.textFormat("%d", .{epoch_label}), @intFromFloat(x - 10), @intFromFloat(graph_rect.y + graph_rect.height + 5), 10, rl.Color.black);

        // Horizontal grid lines (loss)
        rl.drawLine(@intFromFloat(graph_rect.x), @intFromFloat(y), @intFromFloat(graph_rect.x + graph_rect.width), @intFromFloat(y), rl.Color.light_gray);
        const loss_label = min_loss + (max_loss - min_loss) * (@as(f32, num_grid_lines) - fi) / @as(f32, num_grid_lines);
        rl.drawText(rl.textFormat("%.4f", .{loss_label}), @intFromFloat(graph_rect.x - 45), @intFromFloat(y - 5), 10, rl.Color.black);
    }

    // Draw the loss curve
    const color = rl.Color{ .r = 255, .g = 100, .b = 100, .a = 255 }; // Red color for the line

    for (1..history.losses.items.len) |i| {
        const prev_epoch = history.epochs.items[i - 1];
        const curr_epoch = history.epochs.items[i];
        const prev_loss = history.losses.items[i - 1];
        const curr_loss = history.losses.items[i];

        // Convert to screen coordinates
        const prev_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(prev_epoch))) / @as(f32, @floatFromInt(max_epoch));
        const prev_y = graph_rect.y + graph_rect.height - (graph_rect.height * (prev_loss - min_loss)) / (max_loss - min_loss);
        const curr_x = graph_rect.x + (graph_rect.width * @as(f32, @floatFromInt(curr_epoch))) / @as(f32, @floatFromInt(max_epoch));
        const curr_y = graph_rect.y + graph_rect.height - (graph_rect.height * (curr_loss - min_loss)) / (max_loss - min_loss);

        // Draw line segment
        rl.drawLine(@intFromFloat(prev_x), @intFromFloat(prev_y), @intFromFloat(curr_x), @intFromFloat(curr_y), color);

        // Draw small circle at each point
        rl.drawCircle(@intFromFloat(curr_x), @intFromFloat(curr_y), 2, color);
    }
}
