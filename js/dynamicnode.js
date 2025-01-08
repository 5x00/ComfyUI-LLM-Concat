import { app } from "../../../scripts/app.js";

const TypeSlot = {
  Input: 1,
  Output: 2,
};

const TypeSlotEvent = {
  Connect: true,
  Disconnect: false,
};

const _ID = [
  "TriggerToPromptAPI",
  "TriggerToPromptCustom",
  "TriggerToPromptSimple",
];
const _PREFIX = "string";
const _TYPE = "STRING";

app.registerExtension({
  name: "5x00.prompt_plus",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (!_ID.includes(nodeData.name)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
      const me = onNodeCreated?.apply(this);

      if (nodeData.name === "TriggerToPromptCustom") {
        this.addInput("model", "CUSTOMMODEL");
      }
      if (nodeData.name === "TriggerToPromptAPI") {
        this.addInput("model", "APIMODEL");
      }
      this.addInput("prompt", "STRING");

      this.addInput(_PREFIX, _TYPE);
      return me;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (
      slotType,
      slot_idx,
      event,
      link_info,
      node_slot
    ) {
      const me = onConnectionsChange?.apply(this, arguments);

      if (slotType === TypeSlot.Input) {
        if (link_info && event === TypeSlotEvent.Connect) {
          // Get the parent (left-side node) from the link
          const fromNode = this.graph._nodes.find(
            (otherNode) => otherNode.id == link_info.origin_id
          );

          if (fromNode) {
            // Make sure the parent slot type is STRING
            const parent_link = fromNode.outputs[link_info.origin_slot];
            if (parent_link?.type === "STRING" && node_slot.name != "prompt") {
              // Allow only STRING type
              node_slot.type = parent_link.type;
              node_slot.name = `${_PREFIX}_`;
            } else {
              // Disconnect the link if the type is not STRING
              if (node_slot.name != "model" && node_slot.name != "prompt") {
                this.graph.removeLink(link_info.id);
              }
            }
          }
        } else if (event === TypeSlotEvent.Disconnect) {
          if (node_slot.name != "model" && node_slot.name != "prompt") {
            this.removeInput(slot_idx);
          }
        }

        // Track each slot name so we can index the uniques
        let idx = this.inputs.length;
        let slot_tracker = {};
        for (const slot of this.inputs) {
          if (slot.link === null) {
            this.removeInput(idx);
            continue;
          }
          idx += 1;
          const name = slot.name.split("_")[0];
          let count = (slot_tracker[name] || 0) + 1;
          slot_tracker[name] = count;
          if (slot.name != "model" && slot.name != "prompt") {
            slot.name = `${name}_${count}`;
          }
        }

        // Ensure the last slot is a dynamic string input
        let last = this.inputs[this.inputs.length - 1];
        if (last === undefined || last.name != _PREFIX || last.type != _TYPE) {
          this.addInput(_PREFIX, _TYPE);
        }

        // force the node to resize itself for the new/deleted connections
        app.graph.setDirtyCanvas(true, false);

        return me;
      }
    };
    return nodeType;
  },
});
