const std = @import("std");

// Read a CSV file and return the content as a byte array
pub fn read_csv(filename: []const u8) ![]u8 {
    std.debug.print("Opening CSV {s}\n", .{filename});

    // initialize a heap allocator
    const allocator = std.heap.page_allocator;
    // open a file
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    var buffer: [409]u8 = undefined;
    var reader = file.reader(&buffer);
    const reader_interface = &reader.interface;

    var all_data = std.array_list.Managed(u8).init(allocator);
    errdefer all_data.deinit();

    var line_no: usize = 0;
    while (reader_interface.takeDelimiterExclusive('\n')) |line_slice| {
        line_no += 1;
        try all_data.appendSlice(line_slice);
        try all_data.append('\n');
    } else |err| switch (err) {
        error.EndOfStream => {},
        else => return err,
    }

    std.debug.print("Total lines: {d}\n", .{line_no});
    return all_data.toOwnedSlice();
}
