import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const TypeSlot = {
  Input: 1,
  Output: 2,
};

const TypeSlotEvent = {
  Connect: true,
  Disconnect: false,
};

let stringInputs = [];

const _ID = "LLMConcate";
const _PREFIX = "String";
const _TYPE = "STRING";

app.registerExtension({
  name: "5x00.llmconcat",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== _ID) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
      const me = onNodeCreated?.apply(this);
      //Add main widgets
      // Add a single-line text input widget
      this.addWidget(
        "text",
        "API Key",
        "",
        (value) => {
          // You can store the value in a property if needed
          // this.API_Key = value;
        },
        {
          label: "_api", // Label for the widget
          multiline: false, // Disable multiline
        }
      );

      // Add a multi-line text input widget
      this.addWidget(
        "text",
        "Prompt",
        "Generate a image generation prompt that combines {string_1} and {string_2}",
        (value) => {
          // You can store the value in a property if needed
          // this.multilineValue = value;
        },
        {
          label: "prompt", // Label for the widget
          multiline: true, // Enable multiline
        }
      );

      // Add a dropdown menu widget
      this.addWidget(
        "combo",
        "LLM",
        "GPT-4o",
        (value) => {
          // You can store the value in a property if needed
          //this.dropdownValue = value;
        },
        {
          label: "llmSel", // Label for the widget
          values: ["GPT-4o", "Claude"], // Available options
        }
      );

      // start with a new dynamic input specifically for strings
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
            if (parent_link?.type === "STRING") {
              // Allow only STRING type
              node_slot.type = parent_link.type;
              node_slot.name = `${_PREFIX}_`;
            } else {
              // Disconnect the link if the type is not STRING
              this.graph.disconnectLink(link_info.id);
            }
          }
        } else if (event === TypeSlotEvent.Disconnect) {
          this.removeInput(slot_idx);
        }

        // Track each slot name so we can index the uniques
        let idx = Object.keys(this.widgets).length;
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
          slot.name = `${name}_${count}`;
          console.log(slot_tracker[name]);
        }

        // Ensure the last slot is a dynamic string input
        let last = this.inputs[this.inputs.length - 1];
        if (last === undefined || last.name != _PREFIX || last.type != _TYPE) {
          this.addInput(_PREFIX, _TYPE);
        }

        // force the node to resize itself for the new/deleted connections
        this?.graph?.setDirtyCanvas(true);
        return me;
      }
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = async function () {
      // Initialize an object to store dynamic string inputs
      let dynamicStrings = {};
    };
    return nodeType;
  },
});
