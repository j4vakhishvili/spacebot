import { useEffect, useRef, useCallback, useState } from "react";
import { withAuthQuery } from "@/api/client";

type EventHandler = (data: unknown) => void;

export type ConnectionState = "connecting" | "connected" | "reconnecting" | "disconnected";

interface UseEventSourceOptions {
	/** Map of SSE event types to handlers */
	handlers: Record<string, EventHandler>;
	/** Whether to connect (default true) */
	enabled?: boolean;
	/** Called when the connection recovers after a disconnect */
	onReconnect?: () => void;
}

const INITIAL_RETRY_MS = 1000;
const MAX_RETRY_MS = 30_000;
const BACKOFF_MULTIPLIER = 2;

/**
 * SSE hook with exponential backoff, connection state tracking,
 * and reconnect notification for state recovery.
 *
 * On mobile browsers, background tabs often have their SSE connections
 * frozen or dropped. We listen for visibility changes and close the
 * EventSource cleanly on hide, then reconnect immediately on show. This
 * avoids waiting through the exponential backoff and prevents a stale
 * "reconnecting" banner from lingering when the user returns.
 */
export function useEventSource(url: string, options: UseEventSourceOptions) {
	const { handlers, enabled = true, onReconnect } = options;
	const handlersRef = useRef(handlers);
	handlersRef.current = handlers;

	const onReconnectRef = useRef(onReconnect);
	onReconnectRef.current = onReconnect;

	const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");

	const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
	const eventSourceRef = useRef<EventSource | null>(null);
	const retryDelayRef = useRef(INITIAL_RETRY_MS);
	// Tracks whether the current reconnect attempt was triggered by an actual
	// connection error (as opposed to a visibility-change reconnect). This is
	// separate from retryDelayRef because the visibility handler resets the
	// backoff delay when the app returns to the foreground, but we still need
	// to know whether a prior error occurred so onReconnect can re-sync data.
	const pendingErrorReconnectRef = useRef(false);

	const connect = useCallback(() => {
		if (eventSourceRef.current) {
			eventSourceRef.current.close();
		}

		setConnectionState("connecting");

		const source = new EventSource(withAuthQuery(url));
		eventSourceRef.current = source;

		source.onopen = () => {
			// Ignore open events from connections we've already replaced or
			// closed (e.g. via the visibility handler or a newer reconnect
			// attempt). Only the currently active source may update state.
			if (eventSourceRef.current !== source) {
				return;
			}

			// pendingErrorReconnectRef is set in onerror and cleared here after
			// onopen handles it. We use a dedicated flag (instead of checking
			// retryDelayRef) because the visibility handler resets the backoff
			// delay when the app returns to the foreground, and we must still
			// re-sync data if the reconnect is recovering from a prior error.
			const shouldResync = pendingErrorReconnectRef.current;
			pendingErrorReconnectRef.current = false;
			retryDelayRef.current = INITIAL_RETRY_MS;
			setConnectionState("connected");

			if (shouldResync) {
				onReconnectRef.current?.();
			}
		};

		// Register a listener for each event type in handlers
		for (const eventType of Object.keys(handlersRef.current)) {
			source.addEventListener(eventType, (event: MessageEvent) => {
				try {
					const data = JSON.parse(event.data);
					handlersRef.current[eventType]?.(data);
				} catch {
					handlersRef.current[eventType]?.(event.data);
				}
			});
		}

		// Handle the lagged event from the server
		source.addEventListener("lagged", (event: MessageEvent) => {
			try {
				const data = JSON.parse(event.data);
				console.warn(`SSE lagged, skipped ${data.skipped} events`);
			} catch {
				console.warn("SSE lagged, skipped events");
			}
			// Trigger re-sync since we missed events
			onReconnectRef.current?.();
		});

		source.onerror = () => {
			// Ignore errors from connections we've already replaced or closed
			// (e.g. via the visibility handler or a newer reconnect attempt).
			if (eventSourceRef.current !== source) {
				return;
			}

			source.close();
			// Mark that this reconnect is due to an actual error so onopen can
			// decide whether to call onReconnect for missed-event re-sync.
			pendingErrorReconnectRef.current = true;
			setConnectionState("reconnecting");

			const delay = retryDelayRef.current;
			retryDelayRef.current = Math.min(delay * BACKOFF_MULTIPLIER, MAX_RETRY_MS);
			reconnectTimeout.current = setTimeout(connect, delay);
		};
	}, [url]);

	// Close the EventSource cleanly when the tab is hidden so the browser
	// doesn't leave a frozen connection in a half-open state. Reconnect
	// immediately when the tab becomes visible again. State is left as-is
	// on hide; connect() sets "connecting" (hidden by ConnectionBanner when
	// hasData is true) and onopen restores "connected".
	useEffect(() => {
		if (!enabled || typeof document === "undefined") return;

		const handleVisibilityChange = () => {
			if (document.hidden) {
				if (reconnectTimeout.current) {
					clearTimeout(reconnectTimeout.current);
					reconnectTimeout.current = null;
				}
				eventSourceRef.current?.close();
				eventSourceRef.current = null;
			} else {
				retryDelayRef.current = INITIAL_RETRY_MS;
				connect();
			}
		};

		document.addEventListener("visibilitychange", handleVisibilityChange);
		return () => {
			document.removeEventListener("visibilitychange", handleVisibilityChange);
		};
	}, [enabled, connect]);

	useEffect(() => {
		if (!enabled) {
			setConnectionState("disconnected");
			return;
		}

		connect();

		return () => {
			if (reconnectTimeout.current) {
				clearTimeout(reconnectTimeout.current);
			}
			eventSourceRef.current?.close();
		};
	}, [connect, enabled]);

	return { connectionState };
}
