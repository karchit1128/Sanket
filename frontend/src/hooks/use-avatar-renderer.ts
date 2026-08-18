import { useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import type { AnimationContext } from "@/lib/Animations/types";
import { setDefaultPose } from "@/lib/Animations";
import * as AlphabetGestures from "@/lib/Animations";

/**
 * Configuration options for the 3D avatar renderer
 */
interface AvatarRendererConfig {
    containerElement: HTMLElement | null;
    modelPath: string;
    animationSpeed: number;
    pauseDuration: number;
    onTextUpdate?: (text: string) => void;
    onLoadingChange?: (loading: boolean) => void;
}

/**
 * Custom hook for managing Three.js scene and sign language animations
 * Handles 3D avatar rendering, animation queue, and gesture execution
 *
 * @param config - Renderer configuration options
 * @returns Animation control interface
 */
export const useAvatarRenderer = (config: AvatarRendererConfig) => {
    // Store bone transition states for smooth interpolation
    const boneTransitionsRef = useRef<
        Map<
            string,
            {
                startValue: number;
                targetValue: number;
                startTime: number;
                duration: number;
            }
        >
    >(new Map());

    const contextRef = useRef<AnimationContext>({
        animations: [],
        avatar: null,
        pending: false,
        animate: () => {},
        scene: undefined,
        camera: undefined,
        renderer: undefined,
        flag: false,
    });

    const speedRef = useRef(config.animationSpeed);
    const pauseRef = useRef(config.pauseDuration);
    const onTextUpdateRef = useRef(config.onTextUpdate);

    /**
     * Easing function for smooth, human-like motion
     * Uses easeInOutCubic for natural acceleration/deceleration
     */
    const easeInOutCubic = (t: number): number => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    };

    /**
     * Alternative easing: smoother ease-out for more fluid motion
     */
    // const easeOutQuart = (t: number): number => {
    //     return 1 - Math.pow(1 - t, 4);
    // };

    /**
     * Add subtle randomization for organic, human-like variation
     * Humans never move exactly the same way twice
     */
    const addHumanVariation = (
        value: number,
        intensity: number = 0.02,
    ): number => {
        return value * (1 + (Math.random() - 0.5) * intensity);
    };

    // Update animation parameters when config changes
    useEffect(() => {
        speedRef.current = config.animationSpeed;
    }, [config.animationSpeed]);

    useEffect(() => {
        pauseRef.current = config.pauseDuration;
    }, [config.pauseDuration]);

    useEffect(() => {
        onTextUpdateRef.current = config.onTextUpdate;
    }, [config.onTextUpdate]);

    /**
     * Main animation loop - processes queue and updates bone transforms
     */
    const animate = useCallback(() => {
        const ctx = contextRef.current;

        // Exit if queue is empty
        if (ctx.animations.length === 0) {
            ctx.pending = false;
            return;
        }

        requestAnimationFrame(animate);

        // If paused, wait for the pause to complete
        if (ctx.flag) {
            return;
        }

        const currentFrame = ctx.animations[0];

        if (Array.isArray(currentFrame) && currentFrame.length > 0) {
            // Check if this is a text marker (for display synchronization)
            if (currentFrame[0] === "add-text") {
                // Handle text update marker
                if (
                    onTextUpdateRef.current &&
                    typeof currentFrame[1] === "string"
                ) {
                    onTextUpdateRef.current(currentFrame[1]);
                }
                ctx.animations.shift();
                return;
            }

            if (!ctx.flag) {
                const currentTime = performance.now();
                const transitions = boneTransitionsRef.current;

                // Process animation instructions with smooth easing
                /**
                 * Find bone helper that tries multiple naming conventions:
                 * - exact name
                 * - try to insert `:` after `mixamorig` to match rigs like `mixamorig:RightArm`
                 * - fallback: search by suffix (e.g., `RightArm`) as last resort
                 */
                const findBoneByName = (root: any, name: string) => {
                    if (!root) return null;
                    // direct lookup
                    let bone = root.getObjectByName(name);
                    if (bone) return bone;

                    // If we built a boneNameMap on avatar load, prefer using it
                    const map = (ctx as any).boneNameMap as
                        | Record<string, string>
                        | undefined;
                    if (map && map[name]) {
                        bone = root.getObjectByName(map[name]);
                        if (bone) return bone;
                    }

                    // if the name starts with 'mixamorig' but no colon, try adding colon
                    if (name.startsWith("mixamorig") && !name.includes(":")) {
                        const colonName = `mixamorig:${name.substring("mixamorig".length)}`;
                        bone = root.getObjectByName(colonName);
                        if (bone) return bone;
                    }

                    // Fallback: look for bone names that end with suffix (e.g., 'RightArm')
                    const suffix = name.replace(/^.*mixamorig:?/, "");
                    let found: any = null;
                    root.traverse((obj: any) => {
                        if (
                            obj?.name === suffix ||
                            obj?.name?.endsWith?.(suffix)
                        ) {
                            found = obj;
                        }
                    });
                    return found;
                };

                for (let i = 0; i < currentFrame.length; ) {
                    const instruction = currentFrame[i] as [
                        string,
                        string,
                        string,
                        number,
                        string,
                    ];
                    const [boneName, action, axis, targetValue] = instruction;

                    const bone = findBoneByName(ctx.avatar, boneName);

                    if (!bone) {
                        // Remove instruction if bone not found
                        currentFrame.splice(i, 1);
                        continue;
                    }

                    const transitionKey = `${boneName}_${action}_${axis}`;

                    // Initialize transition if not exists
                    if (!transitions.has(transitionKey)) {
                        const currentValue = bone[action][axis];
                        // const distance = Math.abs(targetValue - currentValue);

                        // Calculate duration based on distance and speed
                        // Longer distance = longer duration for consistent motion
                        const baseDuration = 300; // Base 300ms for smooth motion
                        const calculatedDuration = Math.max(
                            200,
                            Math.min(
                                800,
                                baseDuration * (1 / speedRef.current),
                            ),
                        );

                        // Add subtle variation for human-like inconsistency
                        const duration = addHumanVariation(
                            calculatedDuration,
                            0.08,
                        );

                        transitions.set(transitionKey, {
                            startValue: currentValue,
                            targetValue,
                            startTime: currentTime,
                            duration,
                        });
                    }

                    const transition = transitions.get(transitionKey)!;
                    const elapsed = currentTime - transition.startTime;
                    const progress = Math.min(elapsed / transition.duration, 1);

                    // Apply easing function for smooth interpolation
                    const easedProgress = easeInOutCubic(progress);

                    // Calculate new value with easing
                    const range =
                        transition.targetValue - transition.startValue;
                    const newValue =
                        transition.startValue + range * easedProgress;

                    bone[action][axis] = newValue;

                    // Check if animation is complete
                    if (progress >= 1) {
                        // Ensure exact target value
                        bone[action][axis] = targetValue;
                        // Clean up transition
                        transitions.delete(transitionKey);
                        // Remove instruction from queue
                        currentFrame.splice(i, 1);
                    } else {
                        i++;
                    }
                }
            }
        } else {
            // Empty frame - add pause between gestures
            ctx.flag = true;
            setTimeout(() => {
                ctx.flag = false;
            }, pauseRef.current);
            ctx.animations.shift();
        }

        // Render the scene
        if (ctx.renderer && ctx.scene && ctx.camera) {
            ctx.renderer.render(ctx.scene, ctx.camera);
        }
    }, []);

    // Assign animate function to context
    useEffect(() => {
        contextRef.current.animate = animate;
    }, [animate]);

    /**
     * Initialize Three.js scene when container is available
     */
    useEffect(() => {
        if (!config.containerElement) return;

        const ctx = contextRef.current;

        // Reset state
        ctx.flag = false;
        ctx.pending = false;
        ctx.animations = [];

        // Create scene
        ctx.scene = new THREE.Scene();
        ctx.scene.background = new THREE.Color(0xf0f0f0);

        // Enhanced lighting setup for clear material visibility and depth
        // Key light (main light from front-right)
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
        keyLight.position.set(5, 8, 5);
        keyLight.castShadow = true;
        ctx.scene.add(keyLight);

        // Fill light (softer from front-left to reduce harsh shadows)
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
        fillLight.position.set(-5, 5, 5);
        ctx.scene.add(fillLight);

        // Rim/back light for depth and edge definition
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
        rimLight.position.set(0, 3, -5);
        ctx.scene.add(rimLight);

        // Front light specifically for hands/fingers
        const frontLight = new THREE.DirectionalLight(0xffffff, 1.0);
        frontLight.position.set(0, 2, 3);
        ctx.scene.add(frontLight);

        // Ambient light for base illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        ctx.scene.add(ambientLight);

        // Setup camera - positioned for upper body focus
        const containerWidth = config.containerElement.clientWidth;
        const containerHeight = config.containerElement.clientHeight;

        ctx.camera = new THREE.PerspectiveCamera(
            40,
            containerWidth / containerHeight,
            0.1,
            1000,
        );
        // Camera zoomed to chest/hands level for optimal gesture visibility
        ctx.camera.position.set(0, 1.4, 1.5);
        ctx.camera.lookAt(0, 1.3, 0);

        // Setup renderer
        ctx.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
        });
        ctx.renderer.setSize(containerWidth, containerHeight);
        ctx.renderer.setPixelRatio(window.devicePixelRatio);
        ctx.renderer.shadowMap.enabled = true;

        // Add renderer to container (don't clear innerHTML as React manages it)
        config.containerElement.appendChild(ctx.renderer.domElement);

        // Setup Draco loader for compressed models
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath(
            "https://www.gstatic.com/draco/versioned/decoders/1.5.7/",
        );
        dracoLoader.setDecoderConfig({ type: "js" });

        // Load 3D model with Draco support
        const loader = new GLTFLoader();
        loader.setDRACOLoader(dracoLoader);

        console.log("Starting to load model from:", config.modelPath);
        config.onLoadingChange?.(true);

        loader.load(
            config.modelPath,
            (gltf) => {
                console.log("Model loaded successfully, processing meshes...");
                config.onLoadingChange?.(false);

                // Process all meshes in the model
                gltf.scene.traverse((child) => {
                    if (child.type === "SkinnedMesh") {
                        const mesh = child as THREE.SkinnedMesh;
                        mesh.frustumCulled = false;
                        mesh.castShadow = true;
                        mesh.receiveShadow = true;

                        // Fix material properties for better visibility
                        if (mesh.material) {
                            if (Array.isArray(mesh.material)) {
                                mesh.material.forEach((mat) => {
                                    if (
                                        mat instanceof
                                        THREE.MeshStandardMaterial
                                    ) {
                                        mat.roughness = 0.7; // Less shiny, more matte
                                        mat.metalness = 0.1; // Less metallic for natural look
                                        mat.flatShading = false; // Smooth shading
                                        mat.needsUpdate = true;
                                    }
                                });
                            } else if (
                                mesh.material instanceof
                                THREE.MeshStandardMaterial
                            ) {
                                mesh.material.roughness = 0.7;
                                mesh.material.metalness = 0.1;
                                mesh.material.flatShading = false;
                                mesh.material.needsUpdate = true;
                            }
                        }
                    }
                });

                ctx.avatar = gltf.scene;
                ctx.scene!.add(ctx.avatar);

                // Build a bone-name mapping so animation frames can use shorter or older names
                const boneMap: Record<string, string> = {};
                ctx.avatar.traverse((obj: any) => {
                    if (!obj.name) return;
                    boneMap[obj.name] = obj.name;
                    // allow lookup of 'mixamorigRightArm' to 'mixamorig:RightArm'
                    if (obj.name.startsWith("mixamorig:")) {
                        const keyNoColon = obj.name.replace(
                            "mixamorig:",
                            "mixamorig",
                        );
                        boneMap[keyNoColon] = obj.name;
                        const suffix = obj.name.replace(/^.*?:/, "");
                        boneMap[suffix] = obj.name;
                    }
                });
                ctx.boneNameMap = boneMap;

                // Set initial pose
                setDefaultPose(ctx);

                // Start render loop with cleanup support
                if (ctx.renderer && ctx.scene && ctx.camera) {
                    const renderLoop = () => {
                        if (ctx.renderer && ctx.scene && ctx.camera) {
                            const animationId =
                                requestAnimationFrame(renderLoop);
                            // Store animation ID for cleanup
                            (ctx as { renderLoopId?: number }).renderLoopId =
                                animationId;
                            ctx.renderer.render(ctx.scene, ctx.camera);
                        }
                    };
                    renderLoop();
                }

                console.log("✓ Avatar loaded and rendering");
            },
            (progress) => {
                const percent = (progress.loaded / progress.total) * 100;
                console.log(`Loading model: ${percent.toFixed(0)}%`);
            },
            (error) => {
                console.error("Error loading 3D model:", error);
                console.error("Model path attempted:", config.modelPath);
            },
        );

        // Handle window resize
        const handleResize = () => {
            if (!config.containerElement || !ctx.camera || !ctx.renderer)
                return;

            const width = config.containerElement.clientWidth;
            const height = config.containerElement.clientHeight;

            ctx.camera.aspect = width / height;
            ctx.camera.updateProjectionMatrix();
            ctx.renderer.setSize(width, height);
        };

        window.addEventListener("resize", handleResize);

        // Cleanup
        return () => {
            console.log("Starting Three.js cleanup...");

            window.removeEventListener("resize", handleResize);

            // Cancel animation frame
            const renderLoopId = (ctx as { renderLoopId?: number })
                .renderLoopId;
            if (renderLoopId) {
                cancelAnimationFrame(renderLoopId);
            }

            // Cancel any pending animation frames
            ctx.pending = false;

            // Dispose DRACOLoader
            dracoLoader.dispose();

            // Clear scene resources first
            if (ctx.scene) {
                ctx.scene.traverse((object: THREE.Object3D) => {
                    if (object instanceof THREE.Mesh) {
                        object.geometry?.dispose();
                        if (Array.isArray(object.material)) {
                            object.material.forEach((mat) => mat.dispose());
                        } else {
                            object.material?.dispose();
                        }
                    }
                });
            }

            // Dispose renderer and remove canvas
            if (ctx.renderer) {
                const canvas = ctx.renderer.domElement;

                // Dispose renderer first
                ctx.renderer.dispose();

                // Remove canvas from DOM if still attached
                if (canvas && canvas.parentNode) {
                    try {
                        canvas.parentNode.removeChild(canvas);
                    } catch {
                        // Already removed, ignore
                    }
                }
            }

            // Reset context
            ctx.scene = undefined;
            ctx.camera = undefined;
            ctx.renderer = undefined;
            ctx.avatar = null;

            console.log("Three.js cleanup complete");
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [config.containerElement, config.modelPath]);

    /**
     * Execute sign language animation sequence for input text
     * Processes words and individual letters
     */
    const executeSignSequence = useCallback((inputText: string) => {
        const ctx = contextRef.current;

        if (!ctx.avatar) {
            console.warn("Avatar not loaded yet");
            return;
        }

        // Tokenize input: split into words (letters only) and non-words (spaces, punctuation, numbers)
        // This regex matches a sequence of letters OR a sequence of non-letters
        const tokens = inputText.match(/([a-zA-Z]+|[^a-zA-Z]+)/g) || [];

        // A mapping of word variants to canonical TitleCase gesture names
        const wordAliasMap: Record<string, string> = {
            thank: "ThankYou",
            thanks: "ThankYou",
            thankyou: "ThankYou",
            // Add more mappings as needed (e.g., 'thanks' -> 'ThankYou')
        };

        for (let i = 0; i < tokens.length; i++) {
            const token = tokens[i] as string;
            // Check if it's a word (all letters)
            const isWord = /^[a-zA-Z]+$/.test(token);

            if (isWord) {
                // Try to find a whole-word gesture
                // Format: createWordHome, createWordHello (TitleCase)
                // Normalize token to TitleCase. If token has an alias, prefer that canonical name
                const lowerToken = token.toLowerCase();
                const titleCaseWord =
                    wordAliasMap[lowerToken] ||
                    token.charAt(0).toUpperCase() +
                        token.slice(1).toLowerCase();
                const wordFunctionName =
                    `createWord${titleCaseWord}` as keyof typeof AlphabetGestures;
                const wordAnimation = AlphabetGestures[wordFunctionName];

                // Attempt to detect combined two-word gestures like 'thank you'
                // Tokens are pattern: [letters] [non-letters] [letters]
                if (
                    !wordAnimation &&
                    i + 2 < tokens.length &&
                    /^[^a-zA-Z]+$/.test(tokens[i + 1]) &&
                    /^[a-zA-Z]+$/.test(tokens[i + 2])
                ) {
                    const nextWord = tokens[i + 2] as string;
                    const combinedTitleCase =
                        token.charAt(0).toUpperCase() +
                        token.slice(1).toLowerCase() +
                        nextWord.charAt(0).toUpperCase() +
                        nextWord.slice(1).toLowerCase();
                    const combinedFuncName =
                        `createWord${combinedTitleCase}` as keyof typeof AlphabetGestures;
                    const combinedAnimation =
                        AlphabetGestures[combinedFuncName];

                    if (typeof combinedAnimation === "function") {
                        // Start highlight for the first char
                        ctx.animations.push(["add-text", token.charAt(0)]);

                        // Queue combined word animation (e.g., createWordThankYou)
                        combinedAnimation(ctx);

                        // Then show the rest of the combined phrase as text
                        ctx.animations.push([
                            "add-text",
                            token.slice(1) + tokens[i + 1] + nextWord,
                        ]);

                        // Skip the next two tokens since we consumed them
                        i += 3;
                        continue;
                    }
                }

                if (typeof wordAnimation === "function") {
                    // Found a word gesture
                    // Update text with first char to start highlighting
                    ctx.animations.push(["add-text", token.charAt(0)]);

                    // Queue the word animation
                    wordAnimation(ctx);

                    // Update text with the rest of the word after animation starts
                    if (token.length > 1) {
                        ctx.animations.push(["add-text", token.slice(1)]);
                    }
                } else {
                    // No word gesture, spell it out letter by letter
                    for (const char of token) {
                        const upperChar = char.toUpperCase();
                        const letterFunctionName =
                            `createLetter${upperChar}` as keyof typeof AlphabetGestures;
                        const letterAnimation =
                            AlphabetGestures[letterFunctionName];

                        if (typeof letterAnimation === "function") {
                            ctx.animations.push(["add-text", char]);
                            letterAnimation(ctx);
                        } else {
                            // Fallback for unknown letter (shouldn't happen given regex, but safe to handle)
                            ctx.animations.push(["add-text", char]);
                            ctx.animations.push([]); // Pause
                        }
                    }
                }
            } else {
                // Non-word token (spaces, punctuation, numbers)
                // Process character by character with a pause
                for (const char of token) {
                    ctx.animations.push(["add-text", char]);
                    // Add a pause for each non-word character to allow time to "read" it
                    // This solves the "skipping" issue for spaces
                    // Triple pause for spaces/unknowns (3 * 300ms = 900ms)
                    ctx.animations.push([]);
                    ctx.animations.push([]);
                }
            }
        }

        // Start animation loop if not running
        if (!ctx.pending) {
            ctx.pending = true;
            ctx.animate();
        }
    }, []);

    return {
        executeSignSequence,
        context: contextRef.current,
    };
};
