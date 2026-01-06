def rewrite_todo_list() -> None:
    return




def _tool_rewrite_todo_list(self, items: List[str]) -> Dict[str, Any]:
    """Rewrite TODO list with new pending items"""
    # Keep completed and cancelled items
    kept_items = [item for item in self.todo_list if item.status in [TodoStatus.COMPLETED, TodoStatus.CANCELLED]]

    # Create new pending items
    new_items = []
    for content in items:
        new_items.append(TodoItem(id=self.next_todo_id, content=content, status=TodoStatus.PENDING))
        self.next_todo_id += 1

    # Update TODO list
    self.todo_list = kept_items + new_items

    return {"success": True, "kept_items": len(kept_items), "new_items": len(new_items), "total_items": len(self.todo_list)}
